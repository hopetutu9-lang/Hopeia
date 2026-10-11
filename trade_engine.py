"""User-isolated signal execution and protective-position reconciliation."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

from binance_client import CredentialVault, FuturesSession
from config import ADMIN_CHAT_ID
from database import (
    active_trades,
    add_trade,
    all_active_trades,
    daily_realized_pnl,
    get_preference,
    get_setting,
    get_signal,
    get_trade,
    is_premium,
    log_event,
    premium_users_with_preferences,
    record_trade_rejection,
    update_signal_status,
    update_trade,
)
from logger import logger, safe_error_name
from risk_manager import calculate_position, decimal_ceil, decimal_floor


class TradeEngine:
    def __init__(self, application: Any, vault: CredentialVault) -> None:
        self.application = application
        self.vault = vault
        self._monitor_lock = asyncio.Lock()

    async def on_signal(self, signal: dict[str, Any], signal_id: int) -> None:
        stored_signal = get_signal(signal_id) or signal
        users = premium_users_with_preferences()
        for user in users:
            user_id = int(user["user_id"])
            if bool(user["notifications"]):
                await self._notify(
                    user_id,
                    self._format_signal(stored_signal, bool(user["auto_trading"])),
                )
            if bool(user["auto_trading"]):
                await self.execute_for_user(user_id, signal_id)

    async def execute_for_user(self, user_id: int, signal_id: int) -> tuple[bool, str]:
        signal = get_signal(signal_id)
        if not signal:
            return False, "signal not found"
        created_at = signal.get("created_at")
        try:
            created = datetime.fromisoformat(str(created_at))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            max_age = max(
                1,
                min(1440, int(float(get_setting("max_signal_age_minutes", "90") or 90))),
            )
        except (TypeError, ValueError, OverflowError):
            return await self._reject(user_id, signal_id, "API_ERROR")
        if (datetime.now(timezone.utc) - created).total_seconds() > max_age * 60:
            return await self._reject(user_id, signal_id, "SIGNAL_EXPIRED", status="EXPIRED")
        update_signal_status(signal_id, "VALIDATING")
        reason = self._preflight(user_id)
        if reason:
            return await self._reject(user_id, signal_id, reason)

        preference = get_preference(user_id)
        environment = str(preference["environment"])
        if environment not in {"testnet", "real"}:
            return await self._reject(user_id, signal_id, "API_ERROR")
        if environment == "real" and get_setting("real_trading_enabled", "0") != "1":
            return await self._reject(user_id, signal_id, "REAL_TRADING_DISABLED")

        try:
            credentials = self.vault.load_for_user(user_id, environment)
        except Exception:
            return await self._reject(user_id, signal_id, "API_ERROR")
        if not credentials:
            return await self._reject(user_id, signal_id, "API_ERROR")

        api_key, api_secret = credentials
        session = FuturesSession(
            api_key=api_key,
            api_secret=api_secret,
            testnet=environment == "testnet",
        )
        trade_id: int | None = None
        try:
            client = await session.connect()
            await session.verify_trading_access()
            # Always fetch a fresh exchange mark immediately before any sizing
            # or order attempt; scanner candle caches are never execution prices.
            current = await client.futures_mark_price(symbol=signal["symbol"])
            mark_price = float(current.get("markPrice", 0))
            entry = float(signal["entry"])
            stop = float(signal["stop"])
            if not all(value > 0 for value in (mark_price, entry, stop)):
                return await self._reject(user_id, signal_id, "INVALID_PRICE")
            low = float(signal.get("entry_low") or (entry * 0.997))
            high = float(signal.get("entry_high") or (entry * 1.003))
            if low > high or not low <= mark_price <= high:
                return await self._reject(user_id, signal_id, "LIVE_PRICE_OUTSIDE_ENTRY")
            direction = str(signal["direction"])
            target_prices = [float(signal.get(f"tp{index}") or 0) for index in range(1, 4)]
            levels_valid = (
                direction == "LONG"
                and stop < mark_price < target_prices[0] < target_prices[1] < target_prices[2]
            ) or (
                direction == "SHORT"
                and stop > mark_price > target_prices[0] > target_prices[1] > target_prices[2]
            )
            if not levels_valid:
                return await self._reject(user_id, signal_id, "INVALID_PRICE")

            account_equity, available = await session.usdt_equity()
            if account_equity <= 0 or available <= 0:
                return await self._reject(user_id, signal_id, "INSUFFICIENT_BALANCE")
            positions = await client.futures_position_information()
            open_positions = [
                p for p in positions if abs(float(p.get("positionAmt", 0) or 0)) > 0
            ]
            if any(p.get("symbol") == signal["symbol"] for p in open_positions):
                return await self._reject(user_id, signal_id, "EXISTING_POSITION")
            if any(t["symbol"] == signal["symbol"] and t["environment"] == environment for t in active_trades(user_id)):
                return await self._reject(user_id, signal_id, "EXISTING_POSITION")

            daily_pnl = daily_realized_pnl(user_id, environment)
            daily_pnl += sum(
                float(p.get("unRealizedProfit", 0) or 0) for p in open_positions
            )
            info_all = await client.futures_exchange_info()
            symbol_info = next(
                (item for item in info_all.get("symbols", []) if item.get("symbol") == signal["symbol"]),
                None,
            )
            if not symbol_info or symbol_info.get("status") != "TRADING":
                return await self._reject(user_id, signal_id, "API_ERROR")
            min_rr = float(get_setting("min_reward_risk", "1.5") or 1.5)
            risk_distance = abs(mark_price - stop)
            reward = abs(target_prices[0] - mark_price)
            if risk_distance <= 0 or reward / risk_distance < min_rr:
                return await self._reject(user_id, signal_id, "RISK_CHECK_FAILED")
            settings_risk = float(get_setting("risk_per_trade_pct", "0.5") or 0.5)
            user_risk = min(float(preference["risk_pct"]), settings_risk)
            decision = calculate_position(
                equity=account_equity,
                available_balance=available,
                entry=mark_price,
                stop=stop,
                direction=direction,
                risk_pct=user_risk,
                symbol_info=symbol_info,
                daily_realized_pnl=daily_pnl,
                open_position_count=len(open_positions),
            )
            if not decision.approved:
                lower_reason = decision.reason.lower()
                if "quantity" in lower_reason or "notional" in lower_reason or "minimum" in lower_reason:
                    return await self._reject(user_id, signal_id, "INVALID_QUANTITY")
                if "balance" in lower_reason or "margin" in lower_reason:
                    return await self._reject(user_id, signal_id, "INSUFFICIENT_BALANCE")
                return await self._reject(user_id, signal_id, "RISK_CHECK_FAILED")

            trade_id = add_trade(
                user_id,
                signal_id,
                signal["setup_hash"],
                environment,
                signal["symbol"],
                direction,
                str(signal.get("reason") or ""),
            )
            if trade_id is None:
                return await self._reject(user_id, signal_id, "DUPLICATE_SIGNAL")

            leverage = min(3, int(float(get_setting("max_leverage", "3") or 3)))
            await client.futures_change_leverage(symbol=signal["symbol"], leverage=leverage)
            side = "BUY" if direction == "LONG" else "SELL"
            close_side = "SELL" if direction == "LONG" else "BUY"
            client_order_id = f"pro{signal['setup_hash'][:24]}"
            update_signal_status(signal_id, "EXECUTING")
            await self._notify(
                user_id,
                "⚡ EXECUTION\n"
                f"Symbol: {signal['symbol']}\nSide: {direction}\n"
                "🔌 API: CONNECTED\n"
                f"💰 Balance: ${account_equity:,.2f} · Available: ${available:,.2f}\n"
                f"📍 Live Price: {mark_price:g}\n"
                f"🎯 Entry: {low:g} – {high:g}\n"
                "🛡 Risk: PASS\n📊 Position: PASS",
            )
            try:
                entry_order = await client.futures_create_order(
                    symbol=signal["symbol"],
                    side=side,
                    type="MARKET",
                    quantity=decision.quantity,
                    newClientOrderId=client_order_id,
                )
            except Exception as exc:
                # A timeout can occur after Binance accepted the order. Query its
                # deterministic client order ID before deciding the outcome.
                try:
                    entry_order = await client.futures_get_order(
                        symbol=signal["symbol"], origClientOrderId=client_order_id
                    )
                except Exception:
                    update_trade(trade_id, status="UNKNOWN")
                    await self._alert_admin(
                        f"Order state unknown for user {user_id}, {signal['symbol']}. "
                        "Check Binance before retrying."
                    )
                    return await self._reject(user_id, signal_id, "ORDER_REJECTED")
            order_id = str(entry_order.get("orderId", ""))
            status = str(entry_order.get("status", ""))
            if status not in {"FILLED", "PARTIALLY_FILLED"} or not order_id:
                update_trade(trade_id, status="UNKNOWN", entry_order_id=order_id or None)
                return await self._reject(user_id, signal_id, "ORDER_REJECTED")
            if status == "PARTIALLY_FILLED":
                try:
                    await client.futures_cancel_order(symbol=signal["symbol"], orderId=order_id)
                    entry_order = await client.futures_get_order(symbol=signal["symbol"], orderId=order_id)
                except Exception:
                    logger.warning("Could not cancel unfilled remainder trade_id=%s", trade_id)
            filled_qty = float(entry_order.get("executedQty", 0) or 0)
            cumulative = float(entry_order.get("cumQuote", 0) or 0)
            actual_entry = cumulative / filled_qty if filled_qty > 0 and cumulative > 0 else float(entry_order.get("avgPrice", 0) or entry)
            if filled_qty <= 0:
                update_trade(trade_id, status="UNKNOWN", entry_order_id=order_id)
                return await self._reject(user_id, signal_id, "ORDER_REJECTED")

            step, tick = self._steps(symbol_info)
            rounded_stop = (
                decimal_ceil(stop, tick)
                if direction == "LONG"
                else decimal_floor(stop, tick)
            )
            if (
                (direction == "LONG" and rounded_stop >= mark_price)
                or (direction == "SHORT" and rounded_stop <= mark_price)
            ):
                return await self._reject(user_id, signal_id, "INVALID_PRICE")
            update_trade(
                trade_id,
                entry_order_id=order_id,
                exchange_order_ids=json.dumps([order_id]),
                quantity=filled_qty,
                entry_price=actual_entry,
                executed_at=datetime.now(timezone.utc).isoformat(),
                stop_price=rounded_stop,
                best_price=actual_entry,
            )
            try:
                protective = await client.futures_create_order(
                    symbol=signal["symbol"],
                    side=close_side,
                    type="STOP_MARKET",
                    stopPrice=rounded_stop,
                    closePosition="true",
                    workingType="MARK_PRICE",
                    newClientOrderId=f"sl{signal['setup_hash'][:24]}",
                )
                stop_id = str(protective.get("orderId", ""))
                if not stop_id:
                    raise RuntimeError("protective stop did not return an order id")
                checked = await client.futures_get_order(symbol=signal["symbol"], orderId=stop_id)
                if checked.get("status") not in {"NEW", "PARTIALLY_FILLED"}:
                    raise RuntimeError("protective stop was not confirmed open")
                update_trade(
                    trade_id,
                    stop_order_id=stop_id,
                    exchange_order_ids=json.dumps([order_id, stop_id]),
                    status="OPEN",
                )
            except Exception as exc:
                update_trade(trade_id, status="PROTECTION_FAILED")
                closed = await self._emergency_close(
                    client, signal["symbol"], close_side, filled_qty, signal["setup_hash"]
                )
                await self._alert_admin(
                    f"CRITICAL: stop placement failed for user {user_id}, "
                    f"{signal['symbol']}; emergency close {'confirmed' if closed else 'not confirmed'}. "
                    "Manual exchange review is required."
                )
                if closed:
                    update_trade(trade_id, status="CLOSED", closed_at=datetime.now(timezone.utc).isoformat())
                    return await self._reject(user_id, signal_id, "ORDER_REJECTED")
                await self._notify(
                    user_id,
                    f"🚨 CRITICAL: protection failed for {signal['symbol']}. "
                    "Emergency close was not confirmed. Check Binance Futures immediately.",
                )
                update_signal_status(signal_id, "REJECTED", "ORDER_REJECTED")
                return False, "protective order failed; emergency close needs manual review"

            tp_ids = await self._place_take_profits(
                client,
                signal,
                direction,
                close_side,
                filled_qty,
                step,
                tick,
            )
            try:
                tracked_ids = json.loads(
                    get_trade(trade_id).get("exchange_order_ids") or "[]"
                )
            except (ValueError, TypeError, AttributeError):
                tracked_ids = [order_id, stop_id]
            tracked_ids.extend(value for value in tp_ids if value)
            update_trade(
                trade_id,
                tp_order_ids=json.dumps(tp_ids),
                exchange_order_ids=json.dumps(list(dict.fromkeys(tracked_ids))),
            )
            update_signal_status(signal_id, "EXECUTED")
            log_event(
                "trade.opened",
                user_id,
                trade_id=trade_id,
                symbol=signal["symbol"],
                environment=environment,
                qty=filled_qty,
            )
            trade_message = await self._notify(
                user_id,
                f"✅ TRADE EXECUTED\n{direction} {signal['symbol']} · {environment.upper()}\n\n"
                f"Entry: {actual_entry:g}\nQuantity: {filled_qty:g}\nLeverage: {leverage}x\n"
                f"SL: {rounded_stop:g}\nTP1: {target_prices[0]:g}\n"
                f"TP2: {target_prices[1]:g}\nTP3: {target_prices[2]:g}\n"
                f"Order ID: {order_id}\n"
                f"TP orders confirmed: {sum(1 for value in tp_ids if value)}/3\n"
                "Live PnL will refresh on this message.",
            )
            if trade_message is not None:
                update_trade(
                    trade_id,
                    telegram_chat_id=user_id,
                    telegram_message_id=getattr(trade_message, "message_id", None),
                )
            return True, "exchange confirmed"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if trade_id is not None:
                current = active_trades(user_id)
                if any(t["trade_id"] == trade_id and t["status"] == "SUBMITTING" for t in current):
                    update_trade(trade_id, status="UNKNOWN")
            logger.warning(
                "Trade execution rejected user_id=%s signal_id=%s error=%s",
                user_id,
                signal_id,
                safe_error_name(exc),
            )
            return await self._reject(user_id, signal_id, "API_ERROR")
        finally:
            await session.close()

    async def admin_close_trade(self, trade_id: int, admin_id: int) -> tuple[bool, str]:
        """Close one tracked trade after an admin's explicit second confirmation."""
        async with self._monitor_lock:
            trade = get_trade(trade_id)
            if not trade or trade["status"] not in {
                "SUBMITTING",
                "OPEN",
                "PROTECTION_FAILED",
                "UNKNOWN",
            }:
                return False, "Trade is no longer active."
            try:
                credentials = self.vault.load_for_user(
                    int(trade["user_id"]), str(trade["environment"])
                )
            except Exception:
                credentials = None
            if not credentials:
                return False, "Encrypted credentials are unavailable; check the exchange manually."
            session = FuturesSession(
                api_key=credentials[0],
                api_secret=credentials[1],
                testnet=trade["environment"] == "testnet",
            )
            try:
                client = await session.connect()
                positions = await client.futures_position_information(
                    symbol=trade["symbol"]
                )
                position = next(
                    (item for item in positions if item.get("symbol") == trade["symbol"]),
                    None,
                )
                amount = float(position.get("positionAmt", 0) or 0) if position else 0.0
                if amount == 0:
                    await self._cancel_tracked_orders(client, trade)
                    await self._mark_closed(client, trade)
                    log_event(
                        "admin.trade_closed",
                        admin_id,
                        trade_id=trade_id,
                        user_id=trade["user_id"],
                        already_flat=True,
                    )
                    return True, f"Trade #{trade_id} was already flat; tracked orders were cancelled."
                direction_matches = (
                    amount > 0 if trade["direction"] == "LONG" else amount < 0
                )
                if not direction_matches:
                    await self._alert_admin(
                        f"Admin close blocked for trade #{trade_id}: Binance position direction "
                        "does not match the bot record. Review the account manually."
                    )
                    return False, "Exchange position direction differs from the bot record; no order was sent."
                info = await client.futures_exchange_info()
                symbol_info = next(
                    (
                        item
                        for item in info.get("symbols", [])
                        if item.get("symbol") == trade["symbol"]
                    ),
                    {},
                )
                step, _tick = self._steps(symbol_info)
                quantity = decimal_floor(abs(amount), step)
                if quantity <= 0:
                    return False, "Exchange position is below its valid quantity step."
                side = "SELL" if amount > 0 else "BUY"
                close_client_id = (
                    f"adm{trade['setup_hash'][:12]}{int(time.time()) % 100000000:08d}"
                )
                try:
                    close_order = await client.futures_create_order(
                        symbol=trade["symbol"],
                        side=side,
                        type="MARKET",
                        quantity=quantity,
                        reduceOnly="true",
                        newClientOrderId=close_client_id,
                    )
                except Exception as exc:
                    try:
                        close_order = await client.futures_get_order(
                            symbol=trade["symbol"],
                            origClientOrderId=close_client_id,
                        )
                    except Exception:
                        logger.warning(
                            "Admin close outcome unknown trade_id=%s error=%s",
                            trade_id,
                            safe_error_name(exc),
                        )
                        return False, "Close outcome is unknown. Check Binance before retrying."
                order_id = str(close_order.get("orderId", ""))
                if not order_id or close_order.get("status") not in {
                    "FILLED",
                    "PARTIALLY_FILLED",
                }:
                    return False, "Binance did not confirm the close order."
                current_positions = await client.futures_position_information(
                    symbol=trade["symbol"]
                )
                current = next(
                    (
                        item
                        for item in current_positions
                        if item.get("symbol") == trade["symbol"]
                    ),
                    None,
                )
                remainder = (
                    float(current.get("positionAmt", 0) or 0) if current else 0.0
                )
                if abs(remainder) > 0:
                    return (
                        False,
                        f"Binance confirmed a partial close, but {abs(remainder):g} remains open. Review it immediately.",
                    )
                try:
                    tracked_ids = json.loads(trade.get("exchange_order_ids") or "[]")
                except (ValueError, TypeError):
                    tracked_ids = []
                tracked_ids.append(order_id)
                update_trade(
                    trade_id,
                    exchange_order_ids=json.dumps(list(dict.fromkeys(tracked_ids))),
                )
                updated = get_trade(trade_id) or trade
                await self._cancel_tracked_orders(client, updated, except_id=order_id)
                await self._mark_closed(client, updated)
                log_event(
                    "admin.trade_closed",
                    admin_id,
                    trade_id=trade_id,
                    user_id=trade["user_id"],
                    close_order_id=order_id,
                )
                return True, f"Binance confirmed trade #{trade_id} is flat; tracked protection orders were cancelled."
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "Admin close failed trade_id=%s error=%s",
                    trade_id,
                    safe_error_name(exc),
                )
                return False, f"Admin close failed ({safe_error_name(exc)}); check Binance manually."
            finally:
                await session.close()

    async def _cancel_tracked_orders(
        self, client: Any, trade: dict[str, Any], except_id: str | None = None
    ) -> None:
        try:
            order_ids = set(json.loads(trade.get("exchange_order_ids") or "[]"))
            order_ids.update(
                str(value)
                for value in (
                    trade.get("stop_order_id"),
                    *json.loads(trade.get("tp_order_ids") or "[]"),
                )
                if value
            )
        except (TypeError, ValueError):
            order_ids = set()
        for order_id in order_ids:
            if not order_id or str(order_id) == except_id:
                continue
            try:
                order = await client.futures_get_order(
                    symbol=trade["symbol"], orderId=order_id
                )
                if order.get("status") in {"NEW", "PARTIALLY_FILLED"}:
                    await client.futures_cancel_order(
                        symbol=trade["symbol"], orderId=order_id
                    )
            except Exception as exc:
                logger.warning(
                    "Tracked order cancellation unresolved trade_id=%s order_id=%s error=%s",
                    trade["trade_id"],
                    order_id,
                    safe_error_name(exc),
                )

    async def monitor(self) -> None:
        """Reconcile open bot positions and advance protection after TP1."""
        while True:
            try:
                async with self._monitor_lock:
                    for trade in all_active_trades():
                        try:
                            await self._manage_trade(trade)
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            logger.warning(
                                "Position reconciliation failed trade_id=%s error=%s",
                                trade["trade_id"],
                                safe_error_name(exc),
                            )
                await asyncio.sleep(20)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Trade monitor cycle failed error=%s", safe_error_name(exc))
                await asyncio.sleep(20)

    async def _manage_trade(self, trade: dict[str, Any]) -> None:
        try:
            credentials = self.vault.load_for_user(int(trade["user_id"]), trade["environment"])
        except Exception:
            credentials = None
        if not credentials:
            return
        session = FuturesSession(
            api_key=credentials[0],
            api_secret=credentials[1],
            testnet=trade["environment"] == "testnet",
        )
        try:
            client = await session.connect()
            positions = await client.futures_position_information(symbol=trade["symbol"])
            position = next(
                (p for p in positions if p.get("symbol") == trade["symbol"]),
                None,
            )
            amount = float(position.get("positionAmt", 0) or 0) if position else 0.0
            if abs(amount) <= 0:
                close_status, close_reason = await self._confirmed_close_reason(client, trade)
                await self._cancel_tracked_orders(client, trade)
                await self._mark_closed(client, trade, close_status, close_reason)
                return
            if (amount > 0) != (trade["direction"] == "LONG"):
                logger.error(
                    "Exchange position direction mismatch trade_id=%s symbol=%s",
                    trade["trade_id"],
                    trade["symbol"],
                )
                return
            ticker = await client.futures_mark_price(symbol=trade["symbol"])
            mark = float(ticker.get("markPrice", 0) or 0)
            if mark <= 0:
                return
            signal = get_signal(int(trade["signal_id"]))
            if not signal:
                return
            tp_hits = await self._confirmed_take_profit_hits(client, trade)
            if tp_hits[2]:
                if await self._close_tp3_residual(client, trade, amount):
                    return
                # Refresh after a partial close so the displayed quantity and
                # PnL still come from Binance's latest position response.
                positions = await client.futures_position_information(
                    symbol=trade["symbol"]
                )
                position = next(
                    (p for p in positions if p.get("symbol") == trade["symbol"]),
                    None,
                )
                amount = float(position.get("positionAmt", 0) or 0) if position else 0.0
                if abs(amount) <= 0:
                    await self._cancel_tracked_orders(client, trade)
                    await self._mark_closed(
                        client, trade, "COMPLETE", "TP3_CONFIRMED"
                    )
                    return
            await self._maybe_advance_stop(client, trade, mark, tp_hits)
            trade = get_trade(int(trade["trade_id"])) or trade
            entry = float(position.get("entryPrice") or trade.get("entry_price") or 0)
            leverage = float(position.get("leverage") or get_setting("max_leverage", "3") or 1)
            pnl = (
                float(position["unRealizedProfit"])
                if position.get("unRealizedProfit") is not None
                else None
            )
            pnl_percent = (
                pnl / (abs(amount) * entry / leverage) * 100
                if pnl is not None and entry > 0 and leverage > 0
                else None
            )
            live_trade = dict(trade)
            live_trade.update(
                {
                    "entry_price": entry,
                    "mark_price": mark,
                    "position_amount": amount,
                    "quantity": float(trade.get("quantity") or abs(amount)),
                    "remaining_quantity": abs(amount),
                    "leverage": leverage,
                    "unrealized_pnl": pnl,
                    "pnl_percent": pnl_percent,
                    "tp_hits": tp_hits,
                    "stop_price": trade.get("stop_price") or signal.get("stop"),
                    "tp1": signal.get("tp1"),
                    "tp2": signal.get("tp2"),
                    "tp3": signal.get("tp3"),
                }
            )
            await self._update_live_trade_message(live_trade)
        finally:
            await session.close()

    async def _close_tp3_residual(
        self, client: Any, trade: dict[str, Any], amount: float
    ) -> bool:
        """Close any exchange-confirmed remainder after the TP3 order fills."""
        if abs(amount) <= 0:
            return True
        # Cancel remaining limit exits first, but keep the protective stop live
        # until Binance confirms that the position is flat.
        await self._cancel_tracked_orders(
            client, trade, except_id=trade.get("stop_order_id")
        )
        info = await client.futures_exchange_info()
        symbol_info = next(
            (
                item
                for item in info.get("symbols", [])
                if item.get("symbol") == trade["symbol"]
            ),
            {},
        )
        step, _tick = self._steps(symbol_info)
        quantity = decimal_floor(abs(amount), step)
        if quantity <= 0:
            return False
        side = "SELL" if amount > 0 else "BUY"
        client_id = f"t3c{trade['setup_hash'][:16]}{int(time.time()) % 100000000:08d}"
        try:
            order = await client.futures_create_order(
                symbol=trade["symbol"],
                side=side,
                type="MARKET",
                quantity=quantity,
                reduceOnly="true",
                newClientOrderId=client_id[:36],
            )
        except Exception as exc:
            try:
                order = await client.futures_get_order(
                    symbol=trade["symbol"],
                    origClientOrderId=client_id[:36],
                )
            except Exception:
                logger.warning(
                    "TP3 residual close outcome unknown trade_id=%s error=%s",
                    trade["trade_id"],
                    safe_error_name(exc),
                )
                return False
        order_id = str(order.get("orderId", ""))
        if not order_id or order.get("status") not in {
            "FILLED",
            "PARTIALLY_FILLED",
        }:
            logger.warning(
                "TP3 residual close not confirmed trade_id=%s", trade["trade_id"]
            )
            return False
        try:
            order_ids = json.loads(trade.get("exchange_order_ids") or "[]")
        except (TypeError, ValueError):
            order_ids = []
        order_ids.append(order_id)
        update_trade(
            int(trade["trade_id"]),
            exchange_order_ids=json.dumps(list(dict.fromkeys(order_ids))),
        )
        positions = await client.futures_position_information(
            symbol=trade["symbol"]
        )
        current = next(
            (item for item in positions if item.get("symbol") == trade["symbol"]),
            None,
        )
        remaining = float(current.get("positionAmt", 0) or 0) if current else 0.0
        if abs(remaining) > 0:
            logger.warning(
                "TP3 confirmed but residual remains trade_id=%s qty=%s",
                trade["trade_id"],
                abs(remaining),
            )
            return False
        latest = get_trade(int(trade["trade_id"])) or trade
        await self._cancel_tracked_orders(client, latest)
        await self._mark_closed(client, latest, "COMPLETE", "TP3_CONFIRMED")
        return True

    async def _confirmed_take_profit_hits(
        self, client: Any, trade: dict[str, Any]
    ) -> list[bool]:
        try:
            tp_ids = json.loads(trade.get("tp_order_ids") or "[]")
        except (TypeError, ValueError):
            tp_ids = []
        hits = []
        for index in range(3):
            already_hit = bool(trade.get(f"tp{index + 1}_hit"))
            order_id = tp_ids[index] if index < len(tp_ids) else None
            filled = already_hit
            if order_id and not already_hit:
                try:
                    order = await client.futures_get_order(
                        symbol=trade["symbol"], orderId=order_id
                    )
                    filled = order.get("status") == "FILLED"
                except Exception as exc:
                    logger.warning(
                        "Take-profit status check failed trade_id=%s target=%s error=%s",
                        trade["trade_id"],
                        index + 1,
                        safe_error_name(exc),
                    )
            if filled and not already_hit:
                hit_at = datetime.now(timezone.utc).isoformat()
                update_trade(
                    int(trade["trade_id"]),
                    **{f"tp{index + 1}_hit": 1, f"tp{index + 1}_at": hit_at},
                )
                log_event(
                    f"trade.tp{index + 1}_hit",
                    int(trade["user_id"]),
                    trade_id=trade["trade_id"],
                    signal_id=trade["signal_id"],
                    symbol=trade["symbol"],
                    timestamp=hit_at,
                    reason=f"Binance TP{index + 1} order confirmed FILLED",
                )
                trade[f"tp{index + 1}_hit"] = 1
            hits.append(filled)
        return hits

    async def _maybe_advance_stop(
        self, client: Any, trade: dict[str, Any], mark: float, tp_hits: list[bool]
    ) -> None:
        signal = get_signal(int(trade["signal_id"]))
        if not signal:
            return
        entry = float(trade.get("entry_price") or signal["entry"])
        original_stop = float(signal["stop"])
        distance = abs(entry - original_stop)
        if distance <= 0:
            return
        direction = trade["direction"]
        current_stop = float(trade.get("stop_price") or original_stop)
        candidate = current_stop
        if tp_hits and tp_hits[0] and get_setting("move_sl_to_breakeven", "1") == "1":
            candidate = max(candidate, entry) if direction == "LONG" else min(candidate, entry)
        preference = get_preference(int(trade["user_id"]))
        if (
            bool(preference.get("trailing_stop"))
            and get_setting("trailing_stop_enabled", "1") == "1"
        ):
            reached = mark >= entry + 2 * distance if direction == "LONG" else mark <= entry - 2 * distance
            if reached:
                try:
                    trailing_percent = min(
                        10.0,
                        max(0.05, float(get_setting("trailing_stop_percent", "1") or 1)),
                    )
                except (TypeError, ValueError):
                    trailing_percent = 1.0
                trailing = (
                    mark * (1 - trailing_percent / 100)
                    if direction == "LONG"
                    else mark * (1 + trailing_percent / 100)
                )
                candidate = max(candidate, trailing) if direction == "LONG" else min(candidate, trailing)
        improves = candidate > current_stop if direction == "LONG" else candidate < current_stop
        if not improves or (direction == "LONG" and candidate >= mark) or (direction == "SHORT" and candidate <= mark):
            return
        info = await client.futures_exchange_info()
        symbol_info = next(
            (item for item in info.get("symbols", []) if item.get("symbol") == trade["symbol"]),
            {},
        )
        _step, tick = self._steps(symbol_info)
        candidate = decimal_ceil(candidate, tick) if direction == "LONG" else decimal_floor(candidate, tick)
        if (
            (direction == "LONG" and candidate >= mark)
            or (direction == "SHORT" and candidate <= mark)
        ):
            return
        close_side = "SELL" if direction == "LONG" else "BUY"
        new_id = f"mv{trade['setup_hash'][:20]}{int(time.time()) % 100000:05d}"
        # Add and verify the replacement before cancelling the existing stop.
        replacement = await client.futures_create_order(
            symbol=trade["symbol"],
            side=close_side,
            type="STOP_MARKET",
            stopPrice=candidate,
            closePosition="true",
            workingType="MARK_PRICE",
            newClientOrderId=new_id[:36],
        )
        new_order_id = str(replacement.get("orderId", ""))
        if not new_order_id:
            return
        verification = await client.futures_get_order(symbol=trade["symbol"], orderId=new_order_id)
        if verification.get("status") not in {"NEW", "PARTIALLY_FILLED"}:
            return
        old_id = trade.get("stop_order_id")
        if old_id:
            try:
                await client.futures_cancel_order(symbol=trade["symbol"], orderId=old_id)
            except Exception:
                # Keeping both close-position protections is safer than cancelling
                # the replacement after the old order's state became ambiguous.
                logger.warning("Old protective order cancellation unresolved trade_id=%s", trade["trade_id"])
        update_trade(
            int(trade["trade_id"]),
            stop_order_id=new_order_id,
            exchange_order_ids=json.dumps(
                list(
                    dict.fromkeys(
                        json.loads(trade.get("exchange_order_ids") or "[]")
                        + [new_order_id]
                    )
                )
            ),
            stop_price=candidate,
            best_price=mark,
        )
        log_event(
            "trade.stop_advanced",
            int(trade["user_id"]),
            trade_id=trade["trade_id"],
            stop=candidate,
        )

    async def _confirmed_close_reason(
        self, client: Any, trade: dict[str, Any]
    ) -> tuple[str, str]:
        try:
            tp_ids = json.loads(trade.get("tp_order_ids") or "[]")
        except (TypeError, ValueError):
            tp_ids = []
        for index in (2,):
            order_id = tp_ids[index] if len(tp_ids) > index else None
            if order_id:
                try:
                    order = await client.futures_get_order(
                        symbol=trade["symbol"], orderId=order_id
                    )
                    if order.get("status") == "FILLED":
                        return "COMPLETE", "TP3_CONFIRMED"
                except Exception:
                    pass
        stop_id = trade.get("stop_order_id")
        if stop_id:
            try:
                stop_order = await client.futures_get_order(
                    symbol=trade["symbol"], orderId=stop_id
                )
                if stop_order.get("status") == "FILLED":
                    return "STOPPED", "STOP_LOSS_CONFIRMED"
            except Exception:
                pass
        return "CLOSED", "POSITION_CLOSED_WITHOUT_TRACKED_TP3_OR_STOP_FILL"

    async def _mark_closed(
        self,
        client: Any,
        trade: dict[str, Any],
        status: str = "CLOSED",
        reason: str = "POSITION_CLOSED_EXTERNALLY_OR_MANUALLY",
    ) -> None:
        start_ms = 0
        try:
            start_ms = int(datetime.fromisoformat(trade["created_at"]).timestamp() * 1000)
        except (ValueError, TypeError):
            pass
        pnl = 0.0
        pnl_verified = False
        exit_price: float | None = None
        try:
            order_ids = set(json.loads(trade.get("exchange_order_ids") or "[]"))
            order_ids.update(
                str(value)
                for value in (
                    trade.get("entry_order_id"),
                    trade.get("stop_order_id"),
                    *json.loads(trade.get("tp_order_ids") or "[]"),
                )
                if value
            )
            rows = await client.futures_account_trades(
                symbol=trade["symbol"],
                startTime=start_ms,
                limit=1000,
            )
            fills = [item for item in rows if str(item.get("orderId")) in order_ids]
            if fills:
                exit_fills = [
                    item
                    for item in fills
                    if str(item.get("orderId")) != str(trade.get("entry_order_id"))
                ]
                pnl = sum(
                    float(item.get("realizedPnl", 0) or 0)
                    - (
                        float(item.get("commission", 0) or 0)
                        if item.get("commissionAsset") == "USDT"
                        else 0.0
                    )
                    for item in fills
                )
                pnl_verified = bool(exit_fills)
                exit_qty = sum(float(item.get("qty", 0) or 0) for item in exit_fills)
                exit_quote = sum(
                    float(item.get("qty", 0) or 0) * float(item.get("price", 0) or 0)
                    for item in exit_fills
                )
                if exit_qty > 0:
                    exit_price = exit_quote / exit_qty
        except Exception:
            logger.info("Realized PnL reconciliation unavailable trade_id=%s", trade["trade_id"])
        update_trade(
            int(trade["trade_id"]),
            status=status,
            reason=reason,
            close_reason=reason,
            closed_at=datetime.now(timezone.utc).isoformat(),
            exit_price=exit_price,
            realized_pnl=pnl,
            pnl_verified=int(pnl_verified),
        )
        log_event(
            f"trade.{status.lower()}",
            int(trade["user_id"]),
            trade_id=trade["trade_id"],
            signal_id=trade["signal_id"],
            symbol=trade["symbol"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            reason=reason,
            pnl=pnl if pnl_verified else "unavailable",
            pnl_verified=pnl_verified,
        )
        direction_icon = "🟢" if trade["direction"] == "LONG" else "🔴"
        pnl_text = (
            f"Realized PnL (Binance fills, tracked USDT fees): {pnl:+.4f} USDT"
            if pnl_verified
            else "Realized PnL: unavailable (Binance fills could not be reconciled)."
        )
        title = (
            "✅ TRADE COMPLETE" if status == "COMPLETE"
            else "🛑 TRADE STOPPED" if status == "STOPPED"
            else "📕 TRADE CLOSED"
        )
        exit_line = f"Exit: {exit_price:g}\n" if exit_price is not None else ""
        final_text = (
            f"{title}\n{direction_icon} {trade['direction']} {trade['symbol']}\n\n"
            f"Exit reason: {reason}\n{exit_line}"
        )
        await self._edit_trade_message(trade, f"{final_text}{pnl_text}")

    async def _update_live_trade_message(self, trade: dict[str, Any]) -> None:
        direction = str(trade["direction"])
        icon = "🟢" if direction == "LONG" else "🔴"
        pnl = trade.get("unrealized_pnl")
        pnl_text = f"{float(pnl):+,.4f} USDT" if pnl is not None else "Unavailable"
        pnl_percent = trade.get("pnl_percent")
        percent_text = f"{float(pnl_percent):+,.2f}%" if pnl_percent is not None else "Unavailable"
        pnl_status = (
            "🟢 PROFIT" if pnl is not None and float(pnl) > 0
            else "🔴 LOSS" if pnl is not None and float(pnl) < 0
            else "⚪ FLAT" if pnl is not None
            else "⚪ PNL UNAVAILABLE"
        )
        try:
            created = datetime.fromisoformat(str(trade["created_at"]))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            elapsed = max(0, int((datetime.now(timezone.utc) - created).total_seconds()))
            duration = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
        except (ValueError, TypeError):
            duration = "Unavailable"
        hits = trade.get("tp_hits") or [
            bool(trade.get("tp1_hit")),
            bool(trade.get("tp2_hit")),
            bool(trade.get("tp3_hit")),
        ]
        tp_line = " · ".join(
            f"TP{i + 1}{' ✅' if i < len(hits) and hits[i] else ''}: {float(trade.get(f'tp{i + 1}') or 0):g}"
            for i in range(3)
        )
        text = (
            f"📊 LIVE TRADE\n{icon} {direction} {trade['symbol']}\n\n"
            f"Entry: {float(trade.get('entry_price') or 0):g}\n"
            f"Mark Price: {float(trade.get('mark_price') or 0):g}\n"
            f"Quantity: {float(trade.get('quantity') or 0):g} · Remaining: "
            f"{float(trade.get('remaining_quantity') or 0):g}\n"
            f"Leverage: {float(trade.get('leverage') or 0):g}x\n"
            f"Duration: {duration}\n\n"
            f"💰 Unrealized PnL: {pnl_text}\n📈 PnL %: {percent_text}\n\n"
            f"🛑 SL: {float(trade.get('stop_price') or 0):g}\n"
            f"🎯 {tp_line}\n\nStatus: {pnl_status}"
        )
        await self._edit_trade_message(trade, text)

    async def _edit_trade_message(self, trade: dict[str, Any], text: str) -> None:
        chat_id = trade.get("telegram_chat_id") or int(trade["user_id"])
        message_id = trade.get("telegram_message_id")
        if message_id:
            try:
                await self.application.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=text[:3900],
                    disable_web_page_preview=True,
                )
                return
            except Exception:
                logger.info(
                    "Live trade message edit failed trade_id=%s; retrying on next monitor cycle",
                    trade["trade_id"],
                )
                return
        message = await self._notify(int(chat_id), text)
        if message is not None:
            update_trade(
                int(trade["trade_id"]),
                telegram_chat_id=int(chat_id),
                telegram_message_id=getattr(message, "message_id", None),
            )

    async def _place_take_profits(
        self,
        client: Any,
        signal: dict[str, Any],
        direction: str,
        side: str,
        filled_qty: float,
        step: float,
        tick: float,
    ) -> list[str | None]:
        try:
            first_raw = get_setting("tp1_close_percent", "40")
            second_raw = get_setting("tp2_close_percent", "30")
            first = min(
                95.0,
                max(0.0, float("40" if first_raw is None else first_raw)),
            )
            second = min(
                95.0,
                max(0.0, float("30" if second_raw is None else second_raw)),
            )
        except (TypeError, ValueError):
            first, second = 40.0, 30.0
        if first + second > 95:
            scale = 95 / (first + second)
            first, second = first * scale, second * scale
        ratios = (first / 100, second / 100, (100 - first - second) / 100)
        ids: list[str | None] = [None, None, None]
        prices = (float(signal["tp1"]), float(signal["tp2"]), float(signal["tp3"]))
        planned_quantities = [
            decimal_floor(filled_qty * ratio, step) for ratio in ratios
        ]
        planned_quantities[-1] = decimal_floor(
            max(0.0, filled_qty - sum(planned_quantities[:-1])),
            step,
        )
        for index, (ratio, price) in enumerate(zip(ratios, prices), start=1):
            quantity = planned_quantities[index - 1]
            if quantity <= 0:
                continue
            rounded_price = decimal_floor(price, tick) if direction == "LONG" else decimal_ceil(price, tick)
            try:
                order = await client.futures_create_order(
                    symbol=signal["symbol"],
                    side=side,
                    type="LIMIT",
                    timeInForce="GTC",
                    quantity=quantity,
                    price=rounded_price,
                    reduceOnly="true",
                    newClientOrderId=f"tp{index}{signal['setup_hash'][:22]}",
                )
                order_id = str(order.get("orderId", ""))
                if order_id:
                    check = await client.futures_get_order(
                        symbol=signal["symbol"], orderId=order_id
                    )
                    if check.get("status") in {"NEW", "PARTIALLY_FILLED"}:
                        ids[index - 1] = order_id
            except Exception as exc:
                logger.warning(
                    "Take-profit placement failed symbol=%s target=%s error=%s",
                    signal["symbol"],
                    index,
                    safe_error_name(exc),
                )
        return ids

    async def _emergency_close(
        self,
        client: Any,
        symbol: str,
        side: str,
        quantity: float,
        setup_hash: str,
    ) -> bool:
        try:
            order = await client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
                reduceOnly="true",
                newClientOrderId=f"em{setup_hash[:24]}",
            )
            order_id = order.get("orderId")
            if not order_id:
                return False
            checked = await client.futures_get_order(symbol=symbol, orderId=order_id)
            return checked.get("status") in {"FILLED", "PARTIALLY_FILLED"}
        except Exception as exc:
            logger.critical(
                "Emergency close unconfirmed symbol=%s error=%s",
                symbol,
                safe_error_name(exc),
            )
            return False

    def _preflight(self, user_id: int) -> str | None:
        if not is_premium(user_id):
            return "SUBSCRIPTION_INACTIVE"
        if get_setting("emergency_stop", "0") == "1":
            return "EMERGENCY_STOP_ACTIVE"
        if get_setting("global_auto_enabled", "1") != "1":
            return "AUTO_TRADE_DISABLED"
        preference = get_preference(user_id)
        if not preference["auto_trading"]:
            return "AUTO_TRADE_DISABLED"
        return None

    async def _reject(
        self,
        user_id: int,
        signal_id: int,
        reason: str,
        status: str = "REJECTED",
    ) -> tuple[bool, str]:
        signal = get_signal(signal_id) or {}
        update_signal_status(signal_id, status, reason)
        record_trade_rejection(user_id, signal_id, reason)
        timestamp = datetime.now(timezone.utc).isoformat()
        log_event(
            "trade.rejected",
            user_id,
            signal_id=signal_id,
            symbol=signal.get("symbol", "unknown"),
            timestamp=timestamp,
            created_at=signal.get("created_at"),
            reason=reason,
        )
        await self._notify(user_id, f"Trade skipped: {reason}")
        return False, reason

    async def _notify(self, chat_id: int, text: str) -> Any | None:
        try:
            return await self.application.bot.send_message(
                chat_id=chat_id, text=text[:3900], disable_web_page_preview=True
            )
        except Exception:
            logger.info("Telegram notification failed chat_id=%s", chat_id)
            return None

    async def _alert_admin(self, text: str) -> None:
        if ADMIN_CHAT_ID is None:
            return
        try:
            await self.application.bot.send_message(chat_id=ADMIN_CHAT_ID, text=text[:3900])
        except Exception:
            logger.error("Critical admin alert delivery failed")

    @staticmethod
    def _steps(symbol_info: dict[str, Any]) -> tuple[float, float]:
        step = tick = 0.0
        market_step = 0.0
        for item in symbol_info.get("filters", []):
            if item.get("filterType") == "LOT_SIZE":
                step = float(item.get("stepSize", 0) or 0)
            elif item.get("filterType") == "MARKET_LOT_SIZE":
                market_step = float(item.get("stepSize", 0) or 0)
            elif item.get("filterType") == "PRICE_FILTER":
                tick = float(item.get("tickSize", 0) or 0)
        return market_step or step, tick

    @staticmethod
    def _format_signal(signal: dict[str, Any], auto: bool) -> str:
        direction = str(signal.get("direction", "LONG")).upper()
        long_side = direction == "LONG"
        icon = "🟢" if long_side else "🔴"
        entry = float(signal.get("entry") or 0)
        low = float(signal.get("entry_low") or entry * 0.997)
        high = float(signal.get("entry_high") or entry * 1.003)
        stop = float(signal.get("stop") or 0)
        risk = abs(entry - stop)
        ratios = [
            abs(float(signal.get(f"tp{index}") or entry) - entry) / risk
            if risk > 0 else 0
            for index in range(1, 4)
        ]
        created_at = signal.get("created_at")
        age = "00:00"
        try:
            created = datetime.fromisoformat(str(created_at))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            minutes = max(0, int((datetime.now(timezone.utc) - created).total_seconds() // 60))
            age = f"{minutes // 60:02d}:{minutes % 60:02d}"
        except (TypeError, ValueError):
            pass
        structure = str(signal.get("structure") or signal.get("reason") or "SMC structure confirmed")
        confidence = float(signal.get("confidence") or 0)
        timeframe = str(signal.get("timeframe") or "4H / 1H / 5m")
        status = "CHECKING..." if auto else "OFF — REVIEW ONLY"
        return (
            "🚀 SMC TRADE SIGNAL\n━━━━━━━━━━━━━━━━\n"
            f"{icon} {direction} | {signal.get('symbol', 'UNKNOWN')}\n"
            "📊 Binance Futures\n\n"
            f"💰 ENTRY\n{low:g} – {high:g}\n\n"
            f"🛑 STOP LOSS\n{stop:g}\n\n"
            "🎯 TAKE PROFITS\n"
            f"TP1 {float(signal.get('tp1') or 0):g} | {ratios[0]:.1f}R\n"
            f"TP2 {float(signal.get('tp2') or 0):g} | {ratios[1]:.1f}R\n"
            f"TP3 {float(signal.get('tp3') or 0):g} | {ratios[2]:.1f}R\n\n"
            f"📈 STRUCTURE\n{timeframe}: {structure[:500]}\n\n"
            f"🔥 Confidence score: {confidence:.0f}% (strategy confluence, not win probability)\n"
            f"⚖️ RR: {ratios[0]:.1f}R / {ratios[1]:.1f}R / {ratios[2]:.1f}R\n"
            f"⏱ Signal age: {age}\n\n"
            f"🤖 AUTO-EXECUTION: {status}"
        )


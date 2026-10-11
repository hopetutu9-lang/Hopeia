"""Offline safety checks for signal execution and position management.

Run with: python -m unittest discover -s tests -v
All exchange responses are mocked; these tests never place a real order.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import config

_test_root = tempfile.TemporaryDirectory(prefix="binance-bot-tests-")
config.DB_PATH = Path(_test_root.name) / "test.db"
config.LOG_PATH = Path(_test_root.name) / "test.log"

import database
import trade_engine
from trade_engine import TradeEngine


class FakeClient:
    def __init__(self) -> None:
        self.mark = 0.0121
        self.position_amount = 0.0
        self.position_entry = 0.0
        self.position_pnl = 0.0
        self.leverage = 3
        self.next_order_id = 100
        self.orders: dict[str, dict] = {}
        self.order_requests: list[dict] = []
        self.fills: list[dict] = []

    async def futures_mark_price(self, symbol: str) -> dict:
        return {"symbol": symbol, "markPrice": str(self.mark)}

    async def futures_position_information(self, symbol: str | None = None) -> list[dict]:
        if not self.position_amount:
            return []
        return [{
            "symbol": symbol or "ARPAUSDT",
            "positionAmt": str(self.position_amount),
            "entryPrice": str(self.position_entry),
            "markPrice": str(self.mark),
            "unRealizedProfit": str(self.position_pnl),
            "leverage": str(self.leverage),
        }]

    async def futures_exchange_info(self) -> dict:
        return {"symbols": [{
            "symbol": "ARPAUSDT",
            "status": "TRADING",
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "1", "minQty": "1"},
                {"filterType": "MARKET_LOT_SIZE", "stepSize": "1", "minQty": "1"},
                {"filterType": "PRICE_FILTER", "tickSize": "0.0001"},
                {"filterType": "MIN_NOTIONAL", "notional": "5"},
            ],
        }]}

    async def futures_change_leverage(self, symbol: str, leverage: int) -> dict:
        self.leverage = leverage
        return {"symbol": symbol, "leverage": leverage}

    async def futures_create_order(self, **params: object) -> dict:
        self.next_order_id += 1
        order_id = str(self.next_order_id)
        request = dict(params)
        self.order_requests.append(request)
        if request.get("type") == "MARKET":
            quantity = float(request["quantity"])
            if request.get("reduceOnly"):
                if self.position_amount > 0 and request["side"] == "SELL":
                    self.position_amount = max(0.0, self.position_amount - quantity)
                elif self.position_amount < 0 and request["side"] == "BUY":
                    self.position_amount = min(0.0, self.position_amount + quantity)
                self.fills.append({
                    "orderId": order_id,
                    "qty": str(quantity),
                    "price": str(self.mark),
                    "realizedPnl": "3",
                    "commission": "0",
                    "commissionAsset": "USDT",
                })
            else:
                self.position_amount = quantity if request["side"] == "BUY" else -quantity
                self.position_entry = self.mark
            order = {
                **request,
                "orderId": order_id,
                "status": "FILLED",
                "executedQty": str(quantity),
                "cumQuote": str(quantity * self.mark),
                "avgPrice": str(self.mark),
            }
        else:
            order = {**request, "orderId": order_id, "status": "NEW"}
        self.orders[order_id] = order
        return order.copy()

    async def futures_get_order(
        self,
        symbol: str,
        orderId: str | int | None = None,
        origClientOrderId: str | None = None,
    ) -> dict:
        if orderId is not None:
            return self.orders[str(orderId)].copy()
        for order in self.orders.values():
            if order.get("newClientOrderId") == origClientOrderId:
                return order.copy()
        raise LookupError("mock order not found")

    async def futures_cancel_order(self, symbol: str, orderId: str | int) -> dict:
        self.orders[str(orderId)]["status"] = "CANCELED"
        return self.orders[str(orderId)].copy()

    async def futures_account_trades(self, **_params: object) -> list[dict]:
        return list(self.fills)


class FakeSession:
    def __init__(self, client: FakeClient, **_params: object) -> None:
        self.client = client

    async def connect(self) -> FakeClient:
        return self.client

    async def verify_trading_access(self) -> dict:
        return {"can_trade": True, "one_way_mode": True}

    async def usdt_equity(self) -> tuple[float, float]:
        return 10_000.0, 10_000.0

    async def close(self) -> None:
        return None


class FakeVault:
    def load_for_user(self, _user_id: int, _environment: str) -> tuple[str, str]:
        return "offline-test-key", "offline-test-secret"


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.edited: list[dict] = []
        self._message_id = 0

    async def send_message(self, **params: object) -> SimpleNamespace:
        self._message_id += 1
        self.sent.append(dict(params))
        return SimpleNamespace(message_id=self._message_id)

    async def edit_message_text(self, **params: object) -> None:
        self.edited.append(dict(params))


class TradeSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._db_dir = tempfile.TemporaryDirectory(prefix="binance-bot-case-")
        database.DB_PATH = Path(self._db_dir.name) / "case.db"
        database.init_db()
        self.client = FakeClient()
        self.bot = FakeBot()
        self.app = SimpleNamespace(bot=self.bot)
        self.engine = TradeEngine(self.app, FakeVault())
        self.user_id = 912345
        database.upsert_user(self.user_id, "offline-test", "Offline test")
        ok, _message = database.grant_subscription(
            self.user_id, "lifetime", admin_id=self.user_id
        )
        self.assertTrue(ok)
        database.set_preference(self.user_id, "auto_trading", 1)
        database.set_preference(self.user_id, "risk_pct", 0.5)
        database.set_setting("global_auto_enabled", 1)
        database.set_setting("max_signal_age_minutes", 90)
        self._session_patch = patch.object(
            trade_engine,
            "FuturesSession",
            side_effect=lambda **params: FakeSession(self.client, **params),
        )
        self._premium_patch = patch.object(trade_engine, "is_premium", return_value=True)
        self._session_patch.start()
        self._premium_patch.start()

    def tearDown(self) -> None:
        self._premium_patch.stop()
        self._session_patch.stop()
        self._db_dir.cleanup()

    def add_signal(self, direction: str = "LONG") -> int:
        if direction == "LONG":
            stop, tp1, tp2, tp3 = 0.0116, 0.0129, 0.0132, 0.0136
        else:
            stop, tp1, tp2, tp3 = 0.0126, 0.0113, 0.0110, 0.0106
        signal_id = database.store_signal({
            "symbol": "ARPAUSDT",
            "direction": direction,
            "setup_hash": f"offline-{self._db_dir.name[-8:]}-{self._testMethodName}",
            "entry": 0.0121,
            "entry_low": 0.0120,
            "entry_high": 0.0122,
            "zone": {"low": 0.0120, "high": 0.0122},
            "stop": stop,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "confidence": 87,
            "strategy": "SMC",
            "timeframe": "4H / 1H / 5m",
            "structure": "4H HH/HL; 1H sweep + BOS; 5m FVG retest",
            "sweep_time": 1,
            "reason": "offline safety fixture",
        })
        self.assertIsNotNone(signal_id)
        return int(signal_id)

    def execute(self, signal_id: int) -> tuple[bool, str]:
        return asyncio.run(self.engine.execute_for_user(self.user_id, signal_id))

    def test_old_sweep_time_does_not_expire_fresh_signal_and_executes(self) -> None:
        signal_id = self.add_signal()
        signal = database.get_signal(signal_id)
        detected = datetime.fromisoformat(signal["created_at"])
        expires = datetime.fromisoformat(signal["expires_at"])
        self.assertEqual(signal["sweep_time"], 1)
        self.assertEqual(signal["side"], "LONG")
        self.assertEqual(signal["entry_zone"], {"low": 0.012, "high": 0.0122})
        self.assertGreaterEqual((expires - detected).total_seconds(), 90 * 60 - 1)

        ok, reason = self.execute(signal_id)
        self.assertTrue(ok, reason)
        self.assertEqual(database.get_signal(signal_id)["status"], "EXECUTED")
        self.assertTrue(any(order.get("type") == "MARKET" for order in self.client.order_requests))
        self.assertTrue(any(order.get("type") == "STOP_MARKET" for order in self.client.order_requests))
        self.assertEqual(len([o for o in self.client.order_requests if o.get("type") == "LIMIT"]), 3)

    def test_live_price_outside_entry_is_rejected_with_explicit_reason(self) -> None:
        signal_id = self.add_signal()
        self.client.mark = 0.0123
        ok, reason = self.execute(signal_id)
        self.assertFalse(ok)
        self.assertEqual(reason, "LIVE_PRICE_OUTSIDE_ENTRY")
        self.assertFalse(self.client.order_requests)
        self.assertEqual(database.get_signal(signal_id)["reason"], reason)
        history = database.recent_trades(self.user_id, 5)
        self.assertEqual(history[0]["close_reason"], reason)

    def test_only_created_at_controls_ninety_minute_expiry(self) -> None:
        signal_id = self.add_signal()
        stale = (datetime.now(timezone.utc) - timedelta(minutes=91)).isoformat()
        with database.connection() as conn:
            conn.execute(
                "UPDATE signals SET created_at=?, expires_at=? WHERE signal_id=?",
                (stale, stale, signal_id),
            )
        ok, reason = self.execute(signal_id)
        self.assertFalse(ok)
        self.assertEqual(reason, "SIGNAL_EXPIRED")
        self.assertEqual(database.get_signal(signal_id)["status"], "EXPIRED")
        self.assertFalse(self.client.order_requests)

    def test_existing_exchange_position_prevents_duplicate_order(self) -> None:
        signal_id = self.add_signal()
        self.client.position_amount = 20
        self.client.position_entry = 0.012
        ok, reason = self.execute(signal_id)
        self.assertFalse(ok)
        self.assertEqual(reason, "EXISTING_POSITION")
        self.assertFalse(self.client.order_requests)

    def test_live_pnl_tp1_breakeven_and_confirmed_stop_close(self) -> None:
        signal_id = self.add_signal()
        ok, reason = self.execute(signal_id)
        self.assertTrue(ok, reason)
        trade = database.active_trades(self.user_id)[0]
        self.client.position_pnl = 4.82

        asyncio.run(self.engine._manage_trade(trade))
        self.assertTrue(self.bot.edited)
        self.assertIn("+4.8200 USDT", self.bot.edited[-1]["text"])
        self.assertIn("Mark Price: 0.0121", self.bot.edited[-1]["text"])

        tp_ids = json.loads(trade["tp_order_ids"])
        self.client.orders[tp_ids[0]]["status"] = "FILLED"
        self.client.mark = 0.0122
        asyncio.run(self.engine._manage_trade(database.get_trade(trade["trade_id"])))
        updated = database.get_trade(trade["trade_id"])
        self.assertEqual(updated["tp1_hit"], 1)
        self.assertAlmostEqual(updated["stop_price"], updated["entry_price"], places=4)
        self.assertIn("TP1 ✅", self.bot.edited[-1]["text"])
        self.assertIn("SL: 0.0121", self.bot.edited[-1]["text"])

        stop_id = updated["stop_order_id"]
        self.client.orders[stop_id]["status"] = "FILLED"
        self.client.position_amount = 0
        self.client.fills = [{
            "orderId": stop_id,
            "qty": str(updated["quantity"]),
            "price": str(self.client.mark),
            "realizedPnl": "-5",
            "commission": "0.01",
            "commissionAsset": "USDT",
        }]
        asyncio.run(self.engine._manage_trade(updated))
        closed = database.get_trade(trade["trade_id"])
        self.assertEqual(closed["status"], "STOPPED")
        self.assertEqual(closed["close_reason"], "STOP_LOSS_CONFIRMED")
        self.assertEqual(closed["pnl_verified"], 1)
        self.assertAlmostEqual(closed["realized_pnl"], -5.01)
        self.assertIn("TRADE STOPPED", self.bot.edited[-1]["text"])

    def test_confirmed_tp3_closes_exchange_residual_and_completes_trade(self) -> None:
        signal_id = self.add_signal()
        ok, reason = self.execute(signal_id)
        self.assertTrue(ok, reason)
        trade = database.active_trades(self.user_id)[0]
        self.client.mark = 0.0137
        self.client.position_pnl = 9.5
        tp_ids = json.loads(trade["tp_order_ids"])
        self.client.orders[tp_ids[2]]["status"] = "FILLED"

        asyncio.run(self.engine._manage_trade(trade))

        closed = database.get_trade(trade["trade_id"])
        self.assertEqual(self.client.position_amount, 0)
        self.assertEqual(closed["status"], "COMPLETE")
        self.assertEqual(closed["close_reason"], "TP3_CONFIRMED")
        self.assertEqual(closed["tp3_hit"], 1)
        self.assertTrue(any(
            order.get("type") == "MARKET" and order.get("reduceOnly")
            for order in self.client.order_requests
        ))
        self.assertIn("TRADE COMPLETE", self.bot.edited[-1]["text"])

    def test_short_live_pnl_uses_exchange_value_and_red_presentation(self) -> None:
        signal_id = self.add_signal("SHORT")
        ok, reason = self.execute(signal_id)
        self.assertTrue(ok, reason)
        trade = database.active_trades(self.user_id)[0]
        self.assertLess(self.client.position_amount, 0)
        self.client.mark = 0.0122
        self.client.position_pnl = -2.3

        asyncio.run(self.engine._manage_trade(trade))

        message = self.bot.edited[-1]["text"]
        self.assertIn("🔴 SHORT ARPAUSDT", message)
        self.assertIn("-2.3000 USDT", message)
        self.assertIn("🔴 LOSS", message)


if __name__ == "__main__":
    unittest.main()

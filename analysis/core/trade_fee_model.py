from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class TradeFeeModel:
    buy_commission_rate: float = 0.0001
    sell_commission_rate: float = 0.0001
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.0
    legacy_transaction_cost: Optional[float] = None

    @classmethod
    def from_params(
        cls,
        *,
        transaction_cost: Optional[float] = None,
        buy_commission_rate: Optional[float] = None,
        sell_commission_rate: Optional[float] = None,
        min_commission: Optional[float] = None,
        stamp_tax_rate: Optional[float] = None,
        transfer_fee_rate: Optional[float] = None,
    ) -> "TradeFeeModel":
        has_explicit_fee_model = any(
            value is not None
            for value in (buy_commission_rate, sell_commission_rate, min_commission, stamp_tax_rate, transfer_fee_rate)
        )
        legacy = _float_or_none(transaction_cost) if not has_explicit_fee_model else None
        return cls(
            buy_commission_rate=_coalesce_float(buy_commission_rate, legacy, cls.buy_commission_rate),
            sell_commission_rate=_coalesce_float(sell_commission_rate, legacy, cls.sell_commission_rate),
            min_commission=_coalesce_float(min_commission, 0.0 if legacy is not None else cls.min_commission),
            stamp_tax_rate=_coalesce_float(stamp_tax_rate, 0.0 if legacy is not None else cls.stamp_tax_rate),
            transfer_fee_rate=_coalesce_float(transfer_fee_rate, cls.transfer_fee_rate),
            legacy_transaction_cost=legacy,
        )

    def estimate(self, action: str, amount: float) -> dict[str, float]:
        side = str(action or "").lower()
        trade_amount = abs(float(amount or 0.0))
        rate = self.sell_commission_rate if side == "sell" else self.buy_commission_rate
        commission = max(trade_amount * rate, self.min_commission) if trade_amount > 0 else 0.0
        stamp_tax = trade_amount * self.stamp_tax_rate if side == "sell" else 0.0
        transfer_fee = trade_amount * self.transfer_fee_rate
        total_fee = commission + stamp_tax + transfer_fee
        return {
            "commission": commission,
            "stamp_tax": stamp_tax,
            "transfer_fee": transfer_fee,
            "total_fee": total_fee,
        }

    def estimate_round_trip(self, buy_amount: float, sell_amount: float) -> dict[str, float]:
        buy_fee = self.estimate("buy", buy_amount)
        sell_fee = self.estimate("sell", sell_amount)
        return {
            "buy_fee": buy_fee["total_fee"],
            "sell_fee": sell_fee["total_fee"],
            "commission": buy_fee["commission"] + sell_fee["commission"],
            "stamp_tax": sell_fee["stamp_tax"],
            "transfer_fee": buy_fee["transfer_fee"] + sell_fee["transfer_fee"],
            "total_fee": buy_fee["total_fee"] + sell_fee["total_fee"],
        }

    def period_cost_rate(self, turnover: float) -> float:
        if self.legacy_transaction_cost is not None:
            return max(0.0, float(turnover or 0.0)) * self.legacy_transaction_cost
        return max(0.0, float(turnover or 0.0)) * (
            self.buy_commission_rate + self.sell_commission_rate + self.stamp_tax_rate + 2 * self.transfer_fee_rate
        )


def _coalesce_float(*values: Any) -> float:
    for value in values:
        number = _float_or_none(value)
        if number is not None:
            return number
    return 0.0


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None

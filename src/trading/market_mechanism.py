"""
Market mechanism for instant trade matching and execution.
"""

from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime


@dataclass
class BuyOrder:
    """Buy order from a node."""
    node_id: int
    quantity: float
    max_price: float


@dataclass
class SellOrder:
    """Sell order from a node."""
    node_id: int
    quantity: float
    min_price: float


@dataclass
class Transaction:
    """Completed energy transaction."""
    timestamp: datetime
    interval: int
    buyer_id: int
    seller_id: int
    energy_kwh: float
    delivered_kwh: float
    loss_kwh: float
    price_per_kwh: float
    total_cost: float


class MarketMechanism:
    """Handles instant trade matching and execution."""
    
    def __init__(self, config: dict):
        """
        Initialize market mechanism.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.transmission_efficiency = config.get('transmission_efficiency', 0.95)
        self.transactions = []
        
    def match_trades(
        self,
        buy_orders: List[BuyOrder],
        sell_orders: List[SellOrder],
        market_price: float,
        timestamp: datetime,
        interval: int
    ) -> List[Transaction]:
        """
        Match buy and sell orders at market price.
        
        Args:
            buy_orders: List of buy orders
            sell_orders: List of sell orders
            market_price: Current market price
            timestamp: Current timestamp
            interval: Current interval number
            
        Returns:
            List of executed transactions
        """
        transactions = []
        
        valid_buyers = [b for b in buy_orders if b.max_price >= market_price]
        valid_sellers = [s for s in sell_orders if s.min_price <= market_price]
        
        valid_buyers.sort(key=lambda x: x.quantity, reverse=True)
        valid_sellers.sort(key=lambda x: x.quantity, reverse=True)
        
        buyer_idx = 0
        seller_idx = 0
        
        while buyer_idx < len(valid_buyers) and seller_idx < len(valid_sellers):
            buyer = valid_buyers[buyer_idx]
            seller = valid_sellers[seller_idx]
            
            trade_quantity = min(buyer.quantity, seller.quantity)
            
            delivered_quantity = trade_quantity * self.transmission_efficiency
            loss = trade_quantity - delivered_quantity
            
            transaction = Transaction(
                timestamp=timestamp,
                interval=interval,
                buyer_id=buyer.node_id,
                seller_id=seller.node_id,
                energy_kwh=trade_quantity,
                delivered_kwh=delivered_quantity,
                loss_kwh=loss,
                price_per_kwh=market_price,
                total_cost=delivered_quantity * market_price
            )
            
            transactions.append(transaction)
            self.transactions.append(transaction)
            
            buyer.quantity -= trade_quantity
            seller.quantity -= trade_quantity
            
            if buyer.quantity <= 0.01:
                buyer_idx += 1
            if seller.quantity <= 0.01:
                seller_idx += 1
        
        return transactions
    
    def get_transaction_history(self) -> List[Transaction]:
        """Get all transactions executed."""
        return self.transactions
    
    def get_transactions_for_node(self, node_id: int) -> List[Transaction]:
        """Get transactions for a specific node."""
        return [
            t for t in self.transactions
            if t.buyer_id == node_id or t.seller_id == node_id
        ]
    
    def calculate_total_traded(self) -> float:
        """Calculate total energy traded."""
        return sum(t.energy_kwh for t in self.transactions) / 2
    
    def get_market_stats(self) -> Dict:
        """Get market statistics."""
        if not self.transactions:
            return {
                'total_trades': 0,
                'total_energy_traded_kwh': 0.0,
                'avg_trade_size_kwh': 0.0,
                'total_value': 0.0,
                'avg_price': 0.0
            }
        
        return {
            'total_trades': len(self.transactions),
            'total_energy_traded_kwh': self.calculate_total_traded(),
            'avg_trade_size_kwh': sum(t.energy_kwh for t in self.transactions) / len(self.transactions),
            'total_value': sum(t.total_cost for t in self.transactions),
            'avg_price': sum(t.price_per_kwh for t in self.transactions) / len(self.transactions)
        }


if __name__ == "__main__":
    from datetime import datetime
    
    config = {
        'transmission_efficiency': 0.95
    }
    
    market = MarketMechanism(config)
    
    buy_orders = [
        BuyOrder(node_id=1, quantity=1.5, max_price=0.45),
        BuyOrder(node_id=2, quantity=1.0, max_price=0.42),
    ]
    
    sell_orders = [
        SellOrder(node_id=3, quantity=2.0, min_price=0.35),
        SellOrder(node_id=4, quantity=0.8, min_price=0.37),
    ]
    
    market_price = 0.40
    timestamp = datetime(2024, 7, 15, 14, 30)
    
    transactions = market.match_trades(
        buy_orders=buy_orders,
        sell_orders=sell_orders,
        market_price=market_price,
        timestamp=timestamp,
        interval=29
    )
    
    print(f"Executed {len(transactions)} transactions:")
    for t in transactions:
        print(f"  Node {t.seller_id} -> Node {t.buyer_id}: {t.energy_kwh:.2f} kWh at ${t.price_per_kwh:.3f}/kWh")
        print(f"    Delivered: {t.delivered_kwh:.2f} kWh, Loss: {t.loss_kwh:.3f} kWh")
    
    print(f"\nMarket stats:")
    stats = market.get_market_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

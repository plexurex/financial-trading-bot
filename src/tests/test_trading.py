import unittest
from utils.trading_strategy import decide_trade

class TestTradingStrategy(unittest.TestCase):

    def test_decide_trade(self):
        self.assertEqual(decide_trade(0.5, 0.5), "BUY")
        self.assertEqual(decide_trade(-0.5, -0.6), "SELL")
        self.assertEqual(decide_trade(0.0, 0.1), "HOLD")

if __name__ == '__main__':
    unittest.main()

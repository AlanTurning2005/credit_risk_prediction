import numpy as np
import pytest
from src.repro_keras import net_profit

def test_net_profit_hand_calculated():
    """Kiểm tra ca tính tay: 2 khách tốt (0.2, 0.4), 2 khách xấu (0.6, 0.8)."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.2, 0.4, 0.6, 0.8])
    
    # Tại ngưỡng 0.5: Duyệt 2 khách tốt, chặn 2 khách xấu -> Lợi nhuận = 2 * 0.15 = 0.30
    assert net_profit(y_true, y_pred, threshold=0.5) == pytest.approx(0.30)
    
    # Tại ngưỡng 0.7: Duyệt 2 khách tốt, 1 khách xấu lọt vào -> Lợi nhuận = 0.30 - 0.75 = -0.45
    assert net_profit(y_true, y_pred, threshold=0.7) == pytest.approx(-0.45)
    
    # Tại ngưỡng 0.1: Từ chối tất cả -> Lợi nhuận = 0.0
    assert net_profit(y_true, y_pred, threshold=0.1) == pytest.approx(0.0)

    
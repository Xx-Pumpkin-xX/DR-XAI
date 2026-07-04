import numpy as np
import scipy as sp
from functools import partial
from sklearn.metrics import cohen_kappa_score

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """Hàm Loss mô phỏng QWK để scipy tối ưu hóa"""
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
                
        # Trả về âm QWK vì scipy tìm giá trị nhỏ nhất (minimize)
        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        """Tìm ra 4 mốc làm tròn tốt nhất"""
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        # Các mốc khởi tạo ban đầu (giống làm tròn truyền thống)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        
        # Chạy thuật toán Nelder-Mead để tìm mốc tối ưu
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        print(f"   [INFO] Ngưỡng làm tròn tối ưu tìm được: {self.coef_['x']}")

    def predict(self, X, coef):
        """Áp dụng mốc làm tròn để chuyển số thập phân thành class (0-4)"""
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p
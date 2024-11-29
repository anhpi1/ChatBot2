import numpy as np

class DST_block:
    def __init__(self):
        self.Ut = np.array([])  # Khởi tạo Ut là mảng NumPy rỗng
        self.Bt = []
        self.At = 0
        self.Dt = []
        self.DST_history = None

    def update(self, *, Ut=None, Bt=None, At=None, Dt=None, DST_history=None):
        if Ut is not None and len(Ut) > 0:
            self.Ut = np.array(Ut)  # Chuyển đổi Ut thành mảng NumPy
        if Bt is not None:
            self.Bt = Bt
        if At is not None:
            self.At = At
        if Dt is not None:
            self.Dt = Dt
        if DST_history is not None:
            self.DST_history = DST_history

    def __str__(self): 
        return (f"Ut: {np.array_str(self.Ut)}, "
                f"Bt: {self.Bt}, "
                f"At: {self.At}, "
                f"Dt: {self.Dt}, "
                f"DST_history: {self.DST_history}")

# # Khởi tạo đối tượng DST_block
# dst = DST_block()
# dst1 = DST_block()

# # Cập nhật giá trị At với từ khóa
# dst.update(At=3)

# # Cập nhật giá trị Bt với từ khóa
# dst.update(Bt=[1, 2, 3, 4])
# # Cập nhật giá trị At với từ khóa
# dst1.update(At=4)

# # Cập nhật giá trị Bt với từ khóa
# dst1.update(Bt=[5, 6, 7, 8])

# dst.update(DST_history=dst1)

# print(dst)

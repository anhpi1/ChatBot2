from itertools import combinations

# Danh sách ban đầu
numbers = [5, 24, 12, 4, 0, 0, 0, 0, 0]

# Hàm thay thế các phần tử lớn hơn 0
def replace_positive(numbers):
    temp=[]
    # Lấy các chỉ mục của phần tử lớn hơn 0
    positive_indices = [i for i, num in enumerate(numbers) if num > 0]
    
    # Số lượng phần tử lớn hơn 0
    n = len(positive_indices)
    
    # Lặp qua số lượng phần tử thay thế (1 đến n)
    for k in range(1, n + 1):
        # Lấy tất cả các tổ hợp của k phần tử lớn hơn 0
        for positions in combinations(positive_indices, k):
            # Tạo bản sao của danh sách
            new_numbers = numbers[:]
            # Thay thế các vị trí được chọn thành 0
            for pos in positions:
                new_numbers[pos] = 0
            temp.append(new_numbers)
    return temp

# Gọi hàm
print(replace_positive(numbers))


# Danh sách các nhãn
labels = {
    0: "NULL",
    1: "what",
    2: "why",
    3: "when",
    4: "where",
    5: "how",
    6: "how many/ how much",
    7: "yes/no",
    8: "which",
    9: "write/give"
}

# Mở tệp data.txt để đọc và nhan.txt để ghi
with open("temp\data.txt", "r", encoding="utf-8") as data_file, open("temp/nhan.txt", "w", encoding="utf-8") as nhan_file:
    # Đọc từng dòng trong tệp data.txt
    for line in data_file:
        line = line.strip().lower()  # Loại bỏ ký tự thừa và chuyển về chữ thường
        
        # Kiểm tra xem dòng đó có chứa một nhãn nào không
        found_label = None
        for label_id, label_name in labels.items():
            if label_name.lower() in line:  # Kiểm tra xem nhãn có xuất hiện trong dòng
                found_label = label_id
                break
        
        # Nếu tìm thấy nhãn, ghi vào tệp nhan.txt
        if found_label is not None:
            nhan_file.write(f"{found_label}\n")
        else:
            nhan_file.write("0\n")  # Nếu không tìm thấy nhãn, ghi nhãn NULL

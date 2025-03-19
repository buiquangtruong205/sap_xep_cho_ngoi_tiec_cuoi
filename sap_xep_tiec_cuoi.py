import numpy as np

def read_guest_list(file_path):
    with open(file_path, 'r') as file:
        guests = file.read().strip().split(',')
    return [guest.strip() for guest in guests]

# Mã hóa mối quan hệ giữa khách mời
relationship_scores = {
    'vo_chong': 2000,  
    'anh_chi_em': 900,  
    'cha_me_con_cai': 700,  
    'anh_chi_em_ho': 500,  
    'di_chu_bac_chau': 300,  
    'ban_be': 100,  
    'khong_quen': 0  
}

relationship_codes = {
    1: 'vo_chong',  
    2: 'anh_chi_em', 
    3: 'cha_me_con_cai',
    4: 'anh_chi_em_ho',
    5: 'di_chu_bac_chau', 
    6: 'ban_be',
    7: 'khong_quen'
}

# Hàm tính điểm thân thiết của một bàn
def table_score(ban, moi_quan_he):
    diem = 0
    for i in range(len(ban)):
        for j in range(i + 1, len(ban)):
            diem += moi_quan_he[ban[i]].get(ban[j], 0)
    return diem

# Hàm fitness: tính tổng điểm thân thiết của tất cả các bàn
def fitness_function(sap_xep, moi_quan_he, kich_thuoc_ban_toi_da):
    tong_diem = 0
    for ban in sap_xep:
        if len(ban) <= kich_thuoc_ban_toi_da:
            tong_diem += table_score(ban, moi_quan_he)
    return tong_diem

# Khởi tạo quần thể
def initialize_population(kich_thuoc_quan_the, khach_moi, so_ban):
    quan_the = []
    for _ in range(kich_thuoc_quan_the):
        np.random.shuffle(khach_moi)
        sap_xep = [khach_moi[i::so_ban] for i in range(so_ban)]
        quan_the.append(sap_xep)
    return quan_the

# Tính toán độ thích nghi của quần thể
def calculate_fitness(quan_the, moi_quan_he, kich_thuoc_ban_toi_da):
    thich_nghi = [fitness_function(sap_xep, moi_quan_he, kich_thuoc_ban_toi_da) for sap_xep in quan_the]
    return thich_nghi

# Chọn lọc
def selection(quan_the, thich_nghi):
    tong_thich_nghi = np.sum(thich_nghi)
    xac_suat = thich_nghi / tong_thich_nghi
    chi_so = np.random.choice(len(thich_nghi), size=len(thich_nghi), p=xac_suat)
    quan_the_duoc_chon = [quan_the[i] for i in chi_so]
    return quan_the_duoc_chon

# Lai ghép
def crossover(bo_me, ti_le_lai_ghep):
    con_cai = []
    for i in range(0, len(bo_me), 2):
        if i + 1 < len(bo_me) and np.random.rand() < ti_le_lai_ghep:
            diem_cat = np.random.randint(1, len(bo_me[i]))
            con1 = bo_me[i][:diem_cat] + bo_me[i+1][diem_cat:]
            con2 = bo_me[i+1][:diem_cat] + bo_me[i][diem_cat:]
            con_cai.extend([con1, con2])
        else:
            con_cai.extend([bo_me[i], bo_me[i+1]])
    return con_cai

# Đột biến
def mutation(con_cai, ti_le_dot_bien, khach_moi, so_ban):
    for sap_xep in con_cai:
        if np.random.rand() < ti_le_dot_bien:
            ban1, ban2 = np.random.choice(len(sap_xep), 2, replace=False)
            khach1, khach2 = np.random.choice(len(sap_xep[ban1]), 1)[0], np.random.choice(len(sap_xep[ban2]), 1)[0]
            sap_xep[ban1][khach1], sap_xep[ban2][khach2] = sap_xep[ban2][khach2], sap_xep[ban1][khach1]
    return con_cai

# Thuật toán di truyền
def genetic_algorithm(kich_thuoc_quan_the, khach_moi, so_ban, moi_quan_he, kich_thuoc_ban_toi_da, ti_le_lai_ghep, ti_le_dot_bien, so_the_he):
    quan_the = initialize_population(kich_thuoc_quan_the, khach_moi, so_ban)
    best_fitness = []
    avg_fitness = []
    for the_he in range(so_the_he):
        thich_nghi = calculate_fitness(quan_the, moi_quan_he, kich_thuoc_ban_toi_da)
        best_fitness.append(max(thich_nghi))
        avg_fitness.append(np.mean(thich_nghi))
        print(f"Thế hệ {the_he}: Độ thích nghi tốt nhất = {best_fitness[-1]}, Độ thích nghi trung bình = {avg_fitness[-1]}")
        bo_me = selection(quan_the, thich_nghi)
        con_cai = crossover(bo_me, ti_le_lai_ghep)
        quan_the = mutation(con_cai, ti_le_dot_bien, khach_moi, so_ban)
    
    # Tìm sơ đồ sắp xếp tốt nhất
    thich_nghi = calculate_fitness(quan_the, moi_quan_he, kich_thuoc_ban_toi_da)
    chi_so_tot_nhat = np.argmax(thich_nghi)
    sap_xep_tot_nhat = quan_the[chi_so_tot_nhat]
    thich_nghi_tot_nhat = thich_nghi[chi_so_tot_nhat]

    return sap_xep_tot_nhat, thich_nghi_tot_nhat, best_fitness, avg_fitness

# Đọc danh sách khách mời từ file
khach_moi = read_guest_list('dataAI.txt')

# Kích thước tối đa của mỗi bàn
kich_thuoc_ban_toi_da = 6

# Số lượng bàn
so_ban = (len(khach_moi) + kich_thuoc_ban_toi_da - 1) // kich_thuoc_ban_toi_da

# Ví dụ về mối quan hệ giữa các khách mời
loai_moi_quan_he = list(relationship_codes.keys())
moi_quan_he = {khach: {khach_khac: np.random.choice(loai_moi_quan_he) for khach_khac in khach_moi if khach_khac != khach} for khach in khach_moi}

# Chuyển đổi mối quan hệ thành điểm số
moi_quan_he = {khach: {khach_khac: relationship_scores[relationship_codes[moi_quan_he[khach][khach_khac]]] for khach_khac in moi_quan_he[khach]} for khach in moi_quan_he}

# Các tham số của thuật toán di truyền
kich_thuoc_quan_the = 20
ti_le_lai_ghep = 0.8
ti_le_dot_bien = 0.1
so_the_he = 100

# Chạy thuật toán di truyền
sap_xep_tot_nhat, thich_nghi_tot_nhat, best_fitness, avg_fitness = genetic_algorithm(kich_thuoc_quan_the, khach_moi, so_ban, moi_quan_he, kich_thuoc_ban_toi_da, ti_le_lai_ghep, ti_le_dot_bien, so_the_he)

# In ra kết quả
print(f"Điểm thân thiết cao nhất: {thich_nghi_tot_nhat}")

# In ra danh sách khách mời trong từng bàn
for i, ban in enumerate(sap_xep_tot_nhat):
    print(f"Bàn {i+1}: {', '.join(ban)}")

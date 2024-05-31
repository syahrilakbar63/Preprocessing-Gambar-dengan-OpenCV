import cv2
import numpy as np

# Baca gambar dari file
gambar = cv2.imread('gambar_awal.jpg', cv2.IMREAD_COLOR)

# 1. Resizing (Mengubah Ukuran)
ukuran_baru = (200, 200)
gambar_resized = cv2.resize(gambar, ukuran_baru)

# 2. Grayscaling (Konversi ke Grayscale)
gambar_gray = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

# 3. Noise Reduction (Pengurangan Noise)
gambar_blur = cv2.GaussianBlur(gambar_gray, (5, 5), 0)

# 4. Normalization (Normalisasi)
gambar_normalized = cv2.normalize(gambar_blur, None, 0, 255, cv2.NORM_MINMAX)

# 5. Binarization (Konversi ke Hitam-Putih)
_, gambar_binary = cv2.threshold(gambar_normalized, 128, 255, cv2.THRESH_BINARY)

# 6. Contrast Enhancement (Peningkatan Kontras)
gambar_equalized = cv2.equalizeHist(gambar_gray)

# Simpan gambar hasil preprocessing
cv2.imwrite('gambar_resized.jpg', gambar_resized)
cv2.imwrite('gambar_gray.jpg', gambar_gray)
cv2.imwrite('gambar_blur.jpg', gambar_blur)
cv2.imwrite('gambar_normalized.jpg', gambar_normalized)
cv2.imwrite('gambar_binary.jpg', gambar_binary)
cv2.imwrite('gambar_equalized.jpg', gambar_equalized)

print("Preprocessing selesai! Hasil tersimpan dalam file gambar_resized.jpg, gambar_gray.jpg, gambar_blur.jpg, gambar_normalized.jpg, gambar_binary.jpg, dan gambar_equalized.jpg.")

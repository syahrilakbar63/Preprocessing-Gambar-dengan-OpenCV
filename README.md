# Preprocessing Gambar dengan OpenCV

Project ini menggunakan OpenCV untuk melakukan preprocessing gambar. Program membaca gambar input dan menerapkan beberapa teknik preprocessing seperti resizing, grayscaling, pengurangan noise, normalisasi, binarisasi, dan peningkatan kontras. Gambar hasil disimpan sebagai file terpisah untuk analisis lebih lanjut.

## Fitur

- **Resizing**: Mengubah ukuran gambar menjadi 200x200 piksel.
- **Grayscaling**: Mengonversi gambar ke grayscale.
- **Pengurangan Noise**: Mengurangi noise dengan Gaussian blur.
- **Normalisasi**: Menormalkan nilai piksel gambar.
- **Binarisasi**: Mengonversi gambar ke hitam putih dengan threshold.
- **Peningkatan Kontras**: Meningkatkan kontras dengan histogram equalization.

## Cara Penggunaan

1. Letakkan file gambar input bernama `gambar_awal.jpg` di direktori yang sama dengan script.
2. Jalankan script berikut:

    ```python
    python imagePreprocessing.py
    ```

3. Gambar hasil preprocessing akan disimpan dengan nama:
    - `gambar_resized.jpg`
    - `gambar_gray.jpg`
    - `gambar_blur.jpg`
    - `gambar_normalized.jpg`
    - `gambar_binary.jpg`
    - `gambar_equalized.jpg`

## Dependencies

- Python 3.x
- OpenCV (`cv2`)
- NumPy
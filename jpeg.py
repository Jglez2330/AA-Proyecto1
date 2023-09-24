import numpy as np

def quantization(n):
    # This function gives the quantization, 8x8 matrix for a given n
    # n -> Level of quantization, it is an integer between 0 and 100
    Q50 = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 59, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

    if n == 50:
        Q = Q50
    elif n == 0:
        Q = np.ones((8, 8), dtype=int)
    elif 50 < n < 100:
        Q = np.round(((100 - n) / 50) * Q50).astype(int)
    elif 0 < n < 50:
        Q = np.round((50 / n) * Q50).astype(int)
    else:
        Q = None
        print("n must be between 0 and 100")

    return Q

def zigzag(A):
    # This function encodes the matrix A into a vector using the zigzag method
    # A -> matrix
    m, _ = A.shape
    n = int((2 * m - 1) / 2)
    diags = np.flip(np.arange(-n, n + 1))
    s = np.sum(np.abs(A))
    x = []

    for d in diags:
        c = m - abs(d)
        if d >= 0:
            i, j = c, 0
        if d < 0:
            i, j = m, abs(d)

        diag = []
        for _ in range(c):
            diag.append(A[i, j])
            i -= 1
            j += 1

        if abs(d) % 2 == 0:
            x += diag[::-1]
        else:
            x += diag

        s1 = np.sum(np.abs(x))
        if s1 == s:
            break

    return np.array(x)

def izigzag(x, m):
    # This function encodes a vector x into a matrix using the inverse zigzag method
    # x -> zigzag encoded vector
    # m -> size of the final matrix (mxm)
    n = int((2 * m - 1) / 2)
    diags = np.flip(np.arange(-n, n + 1))
    s = np.sum(np.abs(x))
    A = np.zeros((m, m))
    l = 0

    for d in diags:
        c = m - abs(d)
        if d >= 0:
            i, j = c, 0
        if d < 0:
            i, j = m, abs(d)

        elements = x[l:l + c]

        if abs(d) % 2 == 0:
            elements = np.flip(elements)

        for e in range(c):
            A[i, j] = elements[e]
            i -= 1
            j += 1

        l += c

        s1 = np.sum(np.abs(A))
        if s1 == s:
            break

    return A

def jpeg_compression(image, n_l):
    # This function compresses an image into a cell using the JPEG method
    # image -> matrix representing the image
    # n_l -> level of compression of the image
    # compressed_image -> cell representing the 8x8 blocks

    m, n = image.shape
    m = m // 8
    n = n // 8
    compressed_image = np.empty((m, n), dtype=object)

    for i in range(m):
        i_end = 8 * (i + 1)
        i_start = i_end - 8

        for j in range(n):
            j_end = 8 * (j + 1)
            j_start = j_end - 8

            kernel = image[i_start:i_end, j_start:j_end]
            reduced_image = kernel - 128
            dct_image = np.fft.dctn(reduced_image, type=2, norm='ortho')
            Q = quantization(n_l)
            quantized_image = np.round(dct_image / Q).astype(int)
            vect = zigzag(quantized_image)

            compressed_image[i, j] = vect

    return compressed_image

def jpeg_decompression(compressed_image, n_l):
    # This function decompresses an image into a cell using the JPEG method
    # compressed_image -> cell representing the 8x8 blocks of the compressed image
    # n_l -> level of compression of the image
    # decompressed_image -> matrix representing the image

    m, n = compressed_image.shape
    decompressed_image = np.zeros((m * 8, n * 8), dtype=int)

    for i in range(m):
        i_end = 8 * (i + 1)
        i_start = i_end - 8

        for j in range(n):
            j_end = 8 * (j + 1)
            j_start = j_end - 8

            kernel = compressed_image[i, j]
            inverse_zigzag = izigzag(kernel, 8)
            Q = quantization(n_l)
            kernel_image = Q * inverse_zigzag

            rounded_kernel = np.round(np.fft.idctn(kernel_image, type=2, norm='ortho'))
            rounded_kernel = rounded_kernel + 128
            decompressed_image[i_start:i_end, j_start:j_end] = rounded_kernel

    return decompressed_image


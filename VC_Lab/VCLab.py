import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sympy as sp
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import filedialog, messagebox
from scipy import signal
from scipy.fftpack import fft, fftshift, ifft
from skimage import filters
from skimage.io import imshow
from skimage.color import rgb2gray
from scipy.ndimage import convolve
from skimage.io import imread
from scipy.fftpack import fft2, fftshift
from skimage.transform import resize, rotate
from scipy.ndimage import rotate, zoom
from scipy.ndimage import shift
from scipy.integrate import quad  

class MainInterface:
    def __init__(self, master):
        self.master = master
        self.master.title("2024-Visual Computing Lab Experiment")
        self.master.configure(bg='white')

        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()

        window_width = 420
        window_height = 320

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.master.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.lab1_button = tk.Button(master, text="Lab 1", command=self.open_lab1, width=20, height=5)
        self.lab1_button.grid(row=0, column=0, padx=10, pady=10)

        self.lab2_button = tk.Button(master, text="Lab 2", command=self.open_lab2, width=20, height=5)
        self.lab2_button.grid(row=0, column=1, padx=10, pady=10)

        self.lab3_button = tk.Button(master, text="Lab 3", command=self.open_lab3, width=20, height=5)
        self.lab3_button.grid(row=1, column=0, padx=10, pady=10)

        self.lab4_button = tk.Button(master, text="Lab 4", command=self.open_lab4, width=20, height=5)
        self.lab4_button.grid(row=1, column=1, padx=10, pady=10)

        self.exit_button = ttk.Button(master, text="X", command=self.exit_program, style="Exit.TButton")
        self.exit_button.grid(row=2, column=1, sticky="ne", padx=10, pady=10)

        self.style = ttk.Style()
        self.style.configure("Exit.TButton", foreground="red")

    def open_lab1(self):
        self.master.destroy()
        lab1_interface = Lab1Interface()

    def open_lab2(self):
        self.master.destroy()
        lab2_interface = Lab2Interface()

    def open_lab3(self):
        self.master.destroy()
        lab3_interface = Lab3Interface()

    def open_lab4(self):
        self.master.destroy()
        lab4_interface = Lab4Interface()

    def exit_program(self):
        self.master.destroy()


class Lab1Interface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Lab 1")
        self.root.configure(bg='white')

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        window_width = 420
        window_height = 330

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.task1_button = tk.Button(self.root, text="Task 1", command=self.L1task1, width=20, height=5)
        self.task1_button.grid(row=0, column=0, padx=10, pady=10)

        self.task2_button = tk.Button(self.root, text="Task 2", command=self.L1task2, width=20, height=5)
        self.task2_button.grid(row=0, column=1, padx=10, pady=10)

        self.task3_button = tk.Button(self.root, text="Task 3", command=self.L1task3, width=20, height=5)
        self.task3_button.grid(row=1, column=0, padx=10, pady=10)

        self.task4_button = tk.Button(self.root, text="Task 4", command=self.L1task4, width=20, height=5)
        self.task4_button.grid(row=1, column=1, padx=10, pady=10)

        self.back_button = tk.Button(self.root, text="Back to Main Menu", command=self.go_to_main_menu, width=20, height=2)
        self.back_button.grid(row=2, columnspan=2, padx=10, pady=10)

    def L1task1(self):
        # Task 1: Output a 16x16 random image
        random_image = np.random.rand(16, 16)
        plt.imshow(random_image, cmap='gray')
        plt.axis('off')
        plt.show()

    def L1task2(self):
        # Task 2: Color Mode
        image = cv2.imread("lena_color.jpg")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)[1]

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')

        plt.subplot(2, 3, 2)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Grayscale Image')

        plt.subplot(2, 3, 3)
        plt.imshow(binary_image, cmap='gray')
        plt.title('Binary Image')

        r, g, b = cv2.split(image)

        plt.subplot(2, 3, 4)
        plt.imshow(r, cmap='gray')
        plt.title('Red Component')

        plt.subplot(2, 3, 5)
        plt.imshow(g, cmap='gray')
        plt.title('Green Component')

        plt.subplot(2, 3, 6)
        plt.imshow(b, cmap='gray')
        plt.title('Blue Component')

        plt.tight_layout()
        plt.show()

    def L1task3(self):
        # Task 3: Bit image
        gray_image = cv2.imread("flower.jpg", cv2.IMREAD_GRAYSCALE)

        plt.figure(figsize=(10, 8))
        for i in range(8):
            bit_plane = (gray_image >> i) & 1
            plt.subplot(3, 3, i+1)
            plt.imshow(bit_plane, cmap='gray')
            plt.title(f'Bit Plane {i+1}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def L1task4(self):
        # Task 4: Convolution
        image = cv2.imread("flower.jpg", cv2.IMREAD_GRAYSCALE)

        mean_filter = np.ones((3, 3)) / 9
        laplacian_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

        mean_filtered = cv2.filter2D(image, -1, mean_filter)
        laplacian_filtered = cv2.filter2D(image, -1, laplacian_filter)

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(mean_filtered, cmap='gray')
        plt.title('Mean Filtered Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(laplacian_filtered, cmap='gray')
        plt.title('Laplacian Filtered Image')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def go_to_main_menu(self):
        self.root.destroy()
        main_interface = MainInterface(tk.Tk())

class Lab2Interface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Lab 2")
        self.root.configure(bg='white')

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        window_width = 420
        window_height = 320

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        
        self.task1_button = tk.Button(self.root, text="Task 1", command=self.L2task1, width=20, height=5)
        self.task1_button.grid(row=0, column=0, padx=10, pady=10)

        self.task2_button = tk.Button(self.root, text="Task 2", command=self.L2task2, width=20, height=5)
        self.task2_button.grid(row=0, column=1, padx=10, pady=10)

        self.task3_button = tk.Button(self.root, text="Task 3", command=self.L2task3, width=20, height=5)
        self.task3_button.grid(row=1, column=0, padx=10, pady=10)

        self.back_button = tk.Button(self.root, text="Back to Main Menu", command=self.go_to_main_menu, width=20, height=2)
        self.back_button.grid(row=2, columnspan=2, padx=10, pady=10)

    def L2task1(self):
        # Define the list of standard deviations for Gaussian noise
        std_deviations = [0, 0.1, 0.2, 0.5]

        # Define the list of noise densities for salt & pepper noise
        noise_densities = [0, 0.05, 0.1, 0.3]

        # Load image
        image = cv2.imread("lena_color.jpg")
        
        # Create a figure with 2 rows and 4 columns
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Sample Output', fontsize=20)

        # Iterate over each standard deviation for Gaussian noise
        for idx, std_dev in enumerate(std_deviations):
            # Add Gaussian noise
            gaussian_noise = np.random.normal(0, std_dev, image.shape).astype(np.uint8)
            gaussian_image = cv2.add(image, gaussian_noise)
            
            # Plot the Gaussian noise image
            ax = axes[0, idx]
            ax.imshow(cv2.cvtColor(gaussian_image, cv2.COLOR_BGR2RGB))
            ax.set_title(f'std={std_dev}')
            ax.axis('off')
        
        # Iterate over each noise density for salt & pepper noise
        for idx, density in enumerate(noise_densities):
            # Add salt & pepper noise
            salt_pepper_noise = np.zeros_like(image)
            salt_pepper_noise[np.random.rand(*salt_pepper_noise.shape) < density] = 255
            salt_pepper_image = cv2.add(image, salt_pepper_noise)
            
            # Plot the salt & pepper noise image
            ax = axes[1, idx]
            ax.imshow(cv2.cvtColor(salt_pepper_image, cv2.COLOR_BGR2RGB))
            ax.set_title(f'density={density}')
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        
    def L2task2(self):
        class ImageDenoisingApp:
            def __init__(self, master):
                self.master = master
                self.master.title("Image Denoising")

                # GUI Setup
                self.load_button = tk.Button(self.master, text="Load Image", command=self.load_image)
                self.load_button.pack()

                self.methods = ["Averaging", "Gaussian Averaging", "Bilateral Filter", "Median Filter"]
                self.method_buttons = []
                for method in self.methods:
                    button = tk.Button(self.master, text=method, command=lambda m=method: self.apply_filters(m))
                    button.pack()
                    self.method_buttons.append(button)

                self.image_label = tk.Label(self.master)
                self.image_label.pack()

                # Image data
                self.original_image = None
                self.noisy_gaussian = None
                self.noisy_salt_pepper = None

            def load_image(self):
                file_path = filedialog.askopenfilename()
                if file_path:
                    self.original_image = cv2.imread(file_path)
                    if self.original_image is not None:
                        self.generate_noisy_images()
                        self.display_image(self.original_image, title="Original Image")

            def generate_noisy_images(self):
                if self.original_image is not None:
                    self.noisy_gaussian = self.add_gaussian_noise(self.original_image)
                    self.noisy_salt_pepper = self.add_salt_pepper_noise(self.original_image)

            def add_gaussian_noise(self, image):
                mean = 0
                std_dev = 25  # Adjust as needed
                gaussian_noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
                noisy_image = cv2.add(image, gaussian_noise)
                return noisy_image

            def add_salt_pepper_noise(self, image):
                density = 0.02  # Adjust as needed
                salt_pepper_noise = np.zeros_like(image)
                salt_pepper_noise[np.random.rand(*salt_pepper_noise.shape) < density] = 255
                salt_pepper_noise[np.random.rand(*salt_pepper_noise.shape) > 1 - density] = 0
                noisy_image = cv2.add(image, salt_pepper_noise)
                return noisy_image

            def apply_filters(self, method):
                if self.original_image is None:
                    print("Please load an image first.")
                    return

                filter_sizes = [3, 5, 7]
                filtered_images = []

                for size in filter_sizes:
                    if method == "Averaging":
                        filtered_images.append(self.apply_averaging_filter(self.noisy_gaussian, size))
                    elif method == "Gaussian Averaging":
                        filtered_images.append(self.apply_gaussian_averaging_filter(self.noisy_gaussian, size))
                    elif method == "Bilateral Filter":
                        filtered_images.append(self.apply_bilateral_filter(self.noisy_gaussian, size))
                    elif method == "Median Filter":
                        filtered_images.append(self.apply_median_filter(self.noisy_salt_pepper, size))

                self.display_filters_results(self.noisy_gaussian if method != "Median Filter" else self.noisy_salt_pepper, filtered_images, method, filter_sizes)

            def apply_averaging_filter(self, image, size):
                return cv2.blur(image, (size, size))

            def apply_gaussian_averaging_filter(self, image, size):
                return cv2.GaussianBlur(image, (size, size), 0)

            def apply_bilateral_filter(self, image, size):
                return cv2.bilateralFilter(image, size, 75, 75)

            def apply_median_filter(self, image, size):
                return cv2.medianBlur(image, size)

            def display_image(self, image, title="Image"):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(6, 6))
                plt.imshow(image_rgb)
                plt.title(title)
                plt.axis('off')
                plt.show()

            def display_filters_results(self, noisy_image, filtered_images, method, filter_sizes):
                fig, axes = plt.subplots(2, len(filter_sizes) + 1, figsize=(16, 8))
                fig.suptitle(f'{method} operator', fontsize=20)

                axes[0, 0].imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
                axes[0, 0].set_title('Gaussian noise')
                axes[0, 0].axis('off')

                for i, (image, size) in enumerate(zip(filtered_images, filter_sizes)):
                    axes[0, i + 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    axes[0, i + 1].set_title(f'HSIZE={size}')
                    axes[0, i + 1].axis('off')

                axes[1, 0].imshow(cv2.cvtColor(self.noisy_salt_pepper, cv2.COLOR_BGR2RGB))
                axes[1, 0].set_title('salt & pepper noise')
                axes[1, 0].axis('off')

                for i, (image, size) in enumerate(zip(filtered_images, filter_sizes)):
                    axes[1, i + 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    axes[1, i + 1].set_title(f'HSIZE={size}')
                    axes[1, i + 1].axis('off')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()

        root = tk.Tk()
        app = ImageDenoisingApp(root)
        root.mainloop()

        self.original_image = None

    def L2task3(self):
        # Load image
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Failed to load image")
                return

            # Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculate histograms for the grayscale image
            gray_histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

            # Histogram normalization on the grayscale image
            normalized_gray = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
            normalized_gray_histogram = cv2.calcHist([normalized_gray], [0], None, [256], [0, 256])

            # Histogram equalization on the grayscale image
            equalized_gray = cv2.equalizeHist(gray_image)
            equalized_gray_histogram = cv2.calcHist([equalized_gray], [0], None, [256], [0, 256])

            # Process the color image channels
            channels = cv2.split(image)
            normalized_channels = [cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX) for c in channels]
            equalized_channels = [cv2.equalizeHist(c) for c in channels]

            # Merge channels back into a color image
            normalized_color = cv2.merge(normalized_channels)
            equalized_color = cv2.merge(equalized_channels)

            # Prepare data for display
            images = [gray_image, normalized_gray, equalized_gray, image, normalized_color, equalized_color]
            histograms = [gray_histogram, normalized_gray_histogram, equalized_gray_histogram]
            titles = ["Original Grayscale", "Normalized Grayscale", "Equalized Grayscale", "Original Color", "Normalized Color", "Equalized Color"]

            # Display images and histograms
            self.display_histogram(images, histograms, titles)

        else:
            messagebox.showerror("Error", "No image selected")

    def display_histogram(self, images, histograms, titles):
        # Create figure for displaying the images and histograms
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Adjust size as necessary
        
        # Row for grayscale images
        for i in range(3):
            axes[0, i].imshow(images[i], cmap='gray')
            axes[0, i].set_title(titles[i])
            axes[0, i].axis('off')
        
        # Row for histograms
        for i in range(3):
            axes[1, i].plot(histograms[i])
            axes[1, i].set_title(titles[i] + ' Histogram')
            axes[1, i].set_xlim([0, 256])

        # Row for color images
        for i in range(3, 6):
            axes[2, i-3].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            axes[2, i-3].set_title(titles[i])
            axes[2, i-3].axis('off')

        plt.tight_layout()
        plt.show()
    
    def go_to_main_menu(self):
        self.root.destroy()
        main_interface = MainInterface(tk.Tk())


class Lab3Interface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Lab 3")
        self.root.configure(bg='white')

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        window_width = 420
        window_height = 320

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.task1_button = tk.Button(self.root, text="Task 1", command=self.L3task1, width=20, height=5)
        self.task1_button.grid(row=0, column=0, padx=10, pady=10)

        self.task2_button = tk.Button(self.root, text="Task 2", command=self.L3task2, width=20, height=5)
        self.task2_button.grid(row=0, column=1, padx=10, pady=10)
        
        self.back_button = tk.Button(self.root, text="Back to Main Menu", command=self.go_to_main_menu, width=20, height=2)
        self.back_button.grid(row=2, columnspan=2, padx=10, pady=10)

    def go_to_main_menu(self):
        self.root.destroy()
        main_interface = MainInterface(tk.Tk())

    def L3task1(self):
        x = np.linspace(-np.pi, np.pi, 1000)

        def numerical_fourier_series(f, n_terms, x_vals):
            a0 = (1 / np.pi) * quad(lambda x: f(x), -np.pi, np.pi)[0]
            series = a0 / 2
            for n in range(1, n_terms + 1):
                an = (1 / np.pi) * quad(lambda x: f(x) * np.cos(n * x), -np.pi, np.pi)[0]
                bn = (1 / np.pi) * quad(lambda x: f(x) * np.sin(n * x), -np.pi, np.pi)[0]
                series += an * np.cos(n * x_vals) + bn * np.sin(n * x_vals)
            return series

        def plot_all_functions():
            fig, axes = plt.subplots(4, 4, figsize=(15, 10))
            axes = axes.flatten()

            # Define the original function
            f_expr = lambda x: 2 * np.sin(2 * x) + 6 * np.cos(4 * x) + 5 * np.cos(6 * x) + 4 * np.sin(10 * x) + 3 * np.sin(16 * x)
            f_vals = f_expr(x)

            # Plot original function
            axes[0].plot(x, f_vals, label='Original Function')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('f(x)')
            axes[0].legend()
            axes[0].grid(True)

            # Fourier series approximations
            terms_list = [2, 4, 10, 50]
            for i, n_terms in enumerate(terms_list):
                approx_vals = numerical_fourier_series(f_expr, n_terms, x)
                axes[i + 1].plot(x, approx_vals, label=f'Approx. with {n_terms} terms')
                axes[i + 1].set_title(f'{n_terms} terms')
                axes[i + 1].set_xlabel('x')
                axes[i + 1].set_ylabel('f(x)')
                axes[i + 1].legend()
                axes[i + 1].grid(True)

            # Triangular Wave
            triangular_wave = lambda x: np.where((x >= -np.pi) & (x <= np.pi), 1 - abs(x) / np.pi, 0)
            triangular_vals = triangular_wave(x)
            axes[4].plot(x, triangular_vals, label='Triangular Wave')
            axes[4].set_title('Triangular Wave')
            axes[4].set_xlabel('x')
            axes[4].set_ylabel('f(x)')
            axes[4].legend()
            axes[4].grid(True)

            for i, n_terms in enumerate(terms_list):
                tri_approx_vals = numerical_fourier_series(triangular_wave, n_terms, x)
                axes[i + 5].plot(x, tri_approx_vals, label=f'{n_terms} terms')
                axes[i + 5].set_title(f'Triangular {n_terms} terms')
                axes[i + 5].set_xlabel('x')
                axes[i + 5].set_ylabel('f(x)')
                axes[i + 5].legend()
                axes[i + 5].grid(True)

            # Rectangular Wave
            rectangular_wave = lambda x: np.where((x >= -np.pi/2) & (x <= np.pi/2), 1, 0)
            rectangular_vals = rectangular_wave(x)
            axes[9].plot(x, rectangular_vals, label='Rectangular Wave')
            axes[9].set_title('Rectangular Wave')
            axes[9].set_xlabel('x')
            axes[9].set_ylabel('f(x)')
            axes[9].legend()
            axes[9].grid(True)

            for i, n_terms in enumerate(terms_list):
                rect_approx_vals = numerical_fourier_series(rectangular_wave, n_terms, x)
                axes[i + 10].plot(x, rect_approx_vals, label=f'{n_terms} terms')
                axes[i + 10].set_title(f'Rectangular {n_terms} terms')
                axes[i + 10].set_xlabel('x')
                axes[i + 10].set_ylabel('f(x)')
                axes[i + 10].legend()
                axes[i + 10].grid(True)

            # Display the formula in the last plot
            axes[15].axis('off')
            axes[15].set_title('Equation')
            formula_text = r"$f(x) = 2\sin(2x) + 6\cos(4x) + 5\cos(6x) + 4\sin(10x) + 3\sin(16x)$"
            axes[15].text(0.5, 0.5, formula_text, fontsize=14, va='center', ha='center')

            plt.tight_layout()
            plt.show()

        plot_all_functions()

    def L3task2(self):
        x = np.linspace(-np.pi, np.pi, 1000)
        w = np.linspace(-10, 10, 1000)

        def rectangular_wave(x):
            return np.where((x >= -np.pi / 2) & (x <= np.pi / 2), 1, 0)

        def f_cos_3x(x):
            return np.cos(3 * x)

        def f_sin_3x(x):
            return np.sin(3 * x)

        def triangular_wave(x):
            return np.where((x >= -np.pi) & (x <= np.pi), 1 - abs(x) / np.pi, 0)

        def fourier_transform(f, x, w):
            integrand = lambda t, w: f(t) * np.exp(-1j * w * t)
            return np.array([quad(lambda t: np.real(integrand(t, omega)), -np.pi, np.pi)[0] +
                             1j * quad(lambda t: np.imag(integrand(t, omega)), -np.pi, np.pi)[0] for omega in w])

        def fourier_series_coefficients(f, n_terms, x_vals):
            a0 = (1 / np.pi) * quad(lambda x: f(x), -np.pi, np.pi)[0]
            a_n = []
            b_n = []
            for n in range(1, n_terms + 1):
                an = (1 / np.pi) * quad(lambda x: f(x) * np.cos(n * x), -np.pi, np.pi)[0]
                bn = (1 / np.pi) * quad(lambda x: f(x) * np.sin(n * x), -np.pi, np.pi)[0]
                a_n.append(an)
                b_n.append(bn)
            return a0, a_n, b_n

        def plot_function_and_transform(f, f_name, ax1, ax2, ax3):
            f_vals = f(x)
            ax1.plot(x, f_vals, label=f_name)
            ax1.set_xlabel('x')
            ax1.set_ylabel('f(x)')
            ax1.legend()
            ax1.grid(True)

            F_vals = fourier_transform(f, x, w)
            ax2.plot(w, np.abs(F_vals), label=f'Fourier Transform of {f_name}')
            ax2.set_xlabel('w')
            ax2.set_ylabel('|F(w)|')
            ax2.legend()
            ax2.grid(True)

            a0, a_n, b_n = fourier_series_coefficients(f, 10, x)
            n_vals = np.arange(0, 11)
            ax3.stem(n_vals, [a0 / 2] + a_n, label=f'Fourier Coefficients (a_n) of {f_name}', basefmt=" ")
            ax3.stem(n_vals, [0] + b_n, linefmt='r-', markerfmt='ro', label=f'Fourier Coefficients (b_n) of {f_name}', basefmt=" ")
            ax3.set_xlabel('n')
            ax3.set_ylabel('Coefficients')
            ax3.legend()
            ax3.grid(True)

        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        plot_function_and_transform(rectangular_wave, 'Rectangular Wave', axes[0, 0], axes[0, 1], axes[0, 2])
        plot_function_and_transform(f_cos_3x, 'cos(3x)', axes[1, 0], axes[1, 1], axes[1, 2])
        plot_function_and_transform(f_sin_3x, 'sin(3x)', axes[2, 0], axes[2, 1], axes[2, 2])
        plot_function_and_transform(triangular_wave, 'Triangular Wave', axes[3, 0], axes[3, 1], axes[3, 2])

        plt.tight_layout()
        plt.show()

        
class Lab4Interface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Lab 4")
        self.root.configure(bg='white')

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        window_width = 420
        window_height = 320

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.task1_button = tk.Button(self.root, text="Task 1", command=self.L4task1, width=20, height=5)
        self.task1_button.grid(row=0, column=0, padx=10, pady=10)

        self.task2_button = tk.Button(self.root, text="Task 2", command=self.load_image, width=20, height=5)
        self.task2_button.grid(row=0, column=1, padx=10, pady=10)

        self.back_button = tk.Button(self.root, text="Back to Main Menu", command=self.go_to_main_menu, width=20, height=2)
        self.back_button.grid(row=2, columnspan=2, padx=10, pady=10)
        
        self.original_image = None  # Initialize self.original_image


    def go_to_main_menu(self):
        self.root.destroy()
        main_interface = MainInterface(tk.Tk())

    def L4task1(self):
        t = np.linspace(0, 2*np.pi, 2**8)
        y = 10*np.sin(2*t) + 16*np.cos(4*t) + 2*np.cos(t) + 8*np.sin(3*t) + 4*np.sin(50*t)
        
        # Gaussian filter
        mask = np.array([0.05, 0.1, 0.25, 0.3, 0.25, 0.1, 0.05])  # Gaussian filter mask
        y_gaussian = convolve(y, mask, mode='nearest')  # Use a supported mode
        
        # DFT denoise
        y_fft = fft(y)
        u = 6
        y_fft_denoised = y_fft.copy()
        y_fft_denoised[u:len(y_fft)-u] = 0  # Set high-frequency components to zero
        y_dft = np.real(ifft(y_fft_denoised))
        
        # Plot original signal
        plt.figure(figsize=(15, 10))
        plt.subplot(3, 1, 1)
        plt.plot(t, y, label='Original Signal')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title('Original Signal')
        plt.legend()

        # Plot Gaussian denoised signal
        plt.subplot(3, 1, 2)
        plt.plot(t, y_gaussian, label='Denoised Signal (Gaussian Filter)', color='orange')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title('Denoising using Gaussian Filter')
        plt.legend()

        # Plot DFT denoised signal
        plt.subplot(3, 1, 3)
        plt.plot(t, y_dft, label='Denoised Signal (DFT)', color='green')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title('Denoising using DFT')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Combined plot
        plt.figure(figsize=(10, 6))
        plt.plot(t, y, label='Original Signal')
        plt.plot(t, y_gaussian, label='Denoised Signal (Gaussian Filter)', color='orange')
        plt.plot(t, y_dft, label='Denoised Signal (DFT)', color='green')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title('Discrete Fourier Transform')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.original_image is not None:
                self.L4task2()  # Call L4task2 after loading the image
            else:
                messagebox.showerror("Error", "Failed to load image")
        else:
            messagebox.showerror("Error", "No image selected")

    def plot_magnitude_phase(self, img, axes, title):
        F = fft2(img)
        Fshift = fftshift(F)
        magnitude_spectrum = 20 * np.log(np.abs(Fshift))
        phase_spectrum = np.angle(Fshift)
        
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f'Original {title}')
        axes[0].axis('off')
        
        axes[1].imshow(magnitude_spectrum, cmap='gray')
        axes[1].set_title('Magnitude Spectrum')
        axes[1].axis('off')
        
        axes[2].imshow(phase_spectrum, cmap='gray')
        axes[2].set_title('Phase Spectrum')
        axes[2].axis('off')

    def L4task2(self):
        img = self.original_image
        if img is None:
            messagebox.showerror("Error", "No image selected")
            return

        fig, axes = plt.subplots(4, 3, figsize=(15, 20))

        # Original image
        self.plot_magnitude_phase(img, axes[0], 'Image')

        # Shifted image
        shifted_img = np.roll(img, 100, axis=0)
        self.plot_magnitude_phase(shifted_img, axes[1], 'Shifted Image')

        # Scaled image
        scaled_img = zoom(img, 0.5)
        self.plot_magnitude_phase(scaled_img, axes[2], 'Scaled Image')

        # Rotated image
        rotated_img = rotate(img, 90, reshape=False)
        self.plot_magnitude_phase(rotated_img, axes[3], 'Rotated Image')

        plt.tight_layout()
        plt.show()
def main():
    root = tk.Tk()
    main_interface = MainInterface(root)
    root.mainloop()


if __name__ == "__main__":
    main()


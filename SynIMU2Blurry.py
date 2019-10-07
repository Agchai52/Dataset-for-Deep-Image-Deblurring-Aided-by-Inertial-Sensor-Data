import numpy as np
import cv2
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt


class SynImages(object):
    def __init__(self):
        """
        # Parameters

        Sampling Frequency: HZ
        Number of Poses: 30
        Exposure Time: uniform second
        Angular Velocity: uniform rad/second
        Acceleration: uniform m^2/second
        Focal Length: mm
        Pixel Size: m/pixel
        Image Size (Height x Width): pixel

        """
        # Parameter Initialization
        self.sample_freq = 200.
        self.pose = 30

        self.exposure_low = 0.01
        self.exposure_high = 0.1

        self.angular_v_low = -0.1
        self.angular_v_high = 0.1

        self.acceleration_low = -2e-4
        self.acceleration_high = 2e-4

        self.focal_length = 50.
        self.pixel_size = 2.44 * 10 ** -6
        self.image_H, self.image_W = 1280, 720

        self.intrinsicMat = np.array([[self.focal_length/self.pixel_size, 0, self.image_H/2],
                                     [0, self.focal_length/self.pixel_size, self.image_W/2],
                                     [0, 0, 1]])

    # Synthetic Inertial Sensor Data Generation
    def generate_syn_IMU(self):

        # Generate Raw IMU data
        self.exposure = np.random.uniform(low=self.exposure_low, high=self.exposure_high)
        self.interval = self.exposure / self.pose
        self.samples = int(np.floor(self.exposure * self.sample_freq))

        self.gyro_x = 1e-5*np.random.uniform(low=self.angular_v_low, high=self.angular_v_high, size=(self.samples, ))
        self.gyro_y = 1e-5*np.random.uniform(low=self.angular_v_low, high=self.angular_v_high, size=(self.samples, ))
        self.gyro_z = np.random.uniform(low=self.angular_v_low, high=self.angular_v_high, size=(self.samples, ))

        self.acc_x = np.random.uniform(low=self.acceleration_low, high=self.acceleration_high, size=(self.samples, ))
        self.acc_y = np.random.uniform(low=self.acceleration_low, high=self.acceleration_high, size=(self.samples, ))
        self.acc_z = np.random.uniform(low=self.acceleration_low, high=self.acceleration_high, size=(self.samples, ))

        self.raw_gyro_x = np.insert(self.gyro_x, 0, 0.0)
        self.raw_gyro_y = np.insert(self.gyro_y, 0, 0.0)
        self.raw_gyro_z = np.insert(self.gyro_z, 0, 0.0)

        self.raw_acc_x = np.insert(self.acc_x, 0, 0.0)
        self.raw_acc_y = np.insert(self.acc_y, 0, 0.0)
        self.raw_acc_z = np.insert(self.acc_z, 0, 0.0)

        # Interpolation
        # # old timestamp = [0, 1/sample_freq, 2/sample_freq, ..., samples/sample_freq]
        self.old_time_stamp = np.array([i / self.sample_freq for i in range(self.samples+1)])

        # # timestamp = [0, 1*interval, 2*interval, ..., pose*interval]
        self.time_stamp = np.array([i * self.interval for i in range(self.pose+1)])

        self.gyro_x = np.interp(self.time_stamp, self.old_time_stamp, self.raw_gyro_x)
        self.gyro_y = np.interp(self.time_stamp, self.old_time_stamp, self.raw_gyro_y)
        self.gyro_z = np.interp(self.time_stamp, self.old_time_stamp, self.raw_gyro_z)

        self.acc_x = np.array([self.nearest_acc(t, self.raw_acc_x) for t in self.time_stamp])
        self.acc_y = np.array([self.nearest_acc(t, self.raw_acc_y) for t in self.time_stamp])
        self.acc_z = np.array([self.nearest_acc(t, self.raw_acc_z) for t in self.time_stamp])

        #print('exposure =', self.exposure)
        #print('old_timestamp =', self.old_time_stamp)
        #print('timestamp =', self.time_stamp)
        #
        #print('gyro_x =', self.gyro_x)
        #print('gyro_y =', self.gyro_y)
        #print('gyro_z =', self.gyro_z)
        #
        #print('acc_x =', self.acc_x)
        #print('acc_y =', self.acc_y)
        #print('acc_z =', self.acc_z)

    def nearest_acc(self, t, acc):
        """
        :param t: current timestamp: Float
        :param acc: acc of a specific axis: Array([float])
        :return: nearest acc: Float
        """
        i_nearest = np.abs(t-self.old_time_stamp).argmin()
        return acc[i_nearest]


    def compute_rotations(self):
        """
        :return: rotations List[3x3 ndarray]

        dt = exposure / #pose
        R0 =  Identity

        R_i = [[           1, -omega_zi*dt,  omega_yi*dt],
               [ omega_zi*dt,            1, -omega_xi*dt],
               [-omega_yi*dt,  omega_xi*dt,            1]] * R_i-1

        omega_xi = self.gyro_x[i]
        i = 0, 1, 2, ..., pose
        """
        rotations = []
        dt = self.interval
        R = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

        for i in range(self.pose+1):
            omega_xt, omega_yt, omega_zt = self.gyro_x[i], self.gyro_y[i], self.gyro_z[i]
            rotation_operator = np.array([[           1, -omega_zt*dt,  omega_yt*dt],
                                          [ omega_zt*dt,            1, -omega_xt*dt],
                                          [-omega_yt*dt,  omega_xt*dt,            1]])

            new_R = np.matmul(rotation_operator, R)
            R = new_R
            rotations.append(R)

        return rotations


    def compute_translations(self, rotations):
        """
        :param rotations:
        :return translation: List[3x3 ndarray]

        R: rotations

        T_star_0 = [0, 0, 0]^T
        v0 = [0, 0, 0]^T

        a_ = a[i-1] = [self.acc_x[i-1], self.acc_y[i-1], self.acc_z[i-1]]^T
        a  = a[i]   = [self.acc_x[i], self.acc_y[i], self.acc_z[i]]^T
        invR_ = inv(R[i-1]) = np.linalg.inv(rotations[i-1])
        invR  = inv(R[i])   = np.linalg.inv(rotations[i])

        v_i  = v_i-1 + (invR_ * a_ + invR * a) * dt / 2
        T_star_i = T_star_i-1 + (v_i-1 + v_i) * dt / 2
        T_i = T_star_i - R_i * T_star_0
        """
        translations = []
        dt = self.interval
        T0_star = np.array([0, 0, 0]).reshape(3, 1)
        T = np.array([0, 0, 0]).reshape(3, 1)
        T_star = T0_star
        v = np.array([0, 0, 0]).reshape(3, 1)
        translations.append(T)

        for i in range(1, self.pose+1):
            a_ = np.array([self.acc_x[i-1], self.acc_y[i-1], self.acc_z[i-1]]).reshape(3, 1)
            a  = np.array([self.acc_x[i], self.acc_y[i], self.acc_z[i]]).reshape(3, 1)
            invR_ = np.linalg.inv(rotations[i-1])
            invR  = np.linalg.inv(rotations[i])
            R = rotations[i]
            v_ = v

            v = v_ + (np.matmul(invR_, a_) + np.matmul(invR, a)) * dt / 2
            T_star = T_star + (v_ + v) * dt / 2
            T = T_star - np.matmul(R, T0_star)

            translations.append(T)

        return translations

    def syn_homography(self):
        """
        :return: syn_H: List[3x3 Ndarray]


        H_i = K * (R + T * normal_vector ^ T) * inv(K)

        i = 1, 2, 3, ..., pose

        """
        self.generate_syn_IMU()
        rotations = self.compute_rotations()
        translations = self.compute_translations(rotations)

        # Ri3_0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])  # set all rows of column 3 to 0
        norm_v = np.array([0, 0, 1]).reshape(1, 3)  # norm vector of current plane

        K = self.intrinsicMat
        syn_H = []
        for i in range(1, self.pose+1):
            R = rotations[i]  # np.matmul(rotations[i], Ri3_0)
            T = np.matmul(translations[i], norm_v)
            H = np.matmul(np.matmul(K, R+T), np.linalg.inv(K))
            H = H / H[2][2]
            syn_H.append(H)

        return syn_H

    def create_syn_images(self, img, isPlot=False):
        syn_H = self.syn_homography()
        frames = []

        im_src = img
        for i in range(self.pose):
            h_mat = syn_H[i]
            im_dst = cv2.warpPerspective(im_src, h_mat, (self.image_H, self.image_W))
            frames.append(im_dst)
            im_src = im_dst

        frames = np.array(frames)
        blur_img = np.mean(frames, axis=0)
        time_stamp = self.time_stamp
        gyro = np.stack([self.gyro_x, self.gyro_y, self.gyro_z], axis=0)
        acc = np.stack([self.acc_x, self.acc_y, self.acc_z], axis=0)

        #cv2.imshow('Reference', img/255.0)
        #cv2.imshow('Blurry', blur_img/255.0)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        if isPlot:
            self.plot_image_IMU(img, blur_img)
        return blur_img, time_stamp, gyro, acc

    def plot_image_IMU(self, img, blur_img):

        plt.figure()
        plt.plot(self.old_time_stamp, 100000*self.raw_gyro_x, 'or', label='raw_gyro_x')
        plt.plot(self.time_stamp, 100000*self.gyro_x, '-r', label='gyro_x')
        plt.plot(self.old_time_stamp, 100000*self.raw_gyro_y, 'og', label='raw_gyro_y')
        plt.plot(self.time_stamp, 100000*self.gyro_y, '-g', label='gyro_y')
        plt.xlabel("Time / sec")
        plt.ylabel("Angular Velocity * 1e-5 / (rad/s) ")
        plt.title("XY-Axis Gyro Data")
        plt.savefig("Output/plot_XY_Gyro.jpg")

        plt.figure()
        plt.plot(self.old_time_stamp, self.raw_gyro_z, 'ob', label='raw_gyro_z')
        plt.plot(self.time_stamp, self.gyro_z, '-b', label='gyro_z')
        plt.xlabel("Time / sec")
        plt.ylabel("Angular Velocity / (rad/s)")
        plt.title("Z-Axis Gyro Data")
        plt.savefig("Output/plot_Z_Gyro.jpg")

        plt.figure()
        plt.plot(self.old_time_stamp, 1000*self.raw_acc_x, 'or', label='raw_acc_x')
        plt.plot(self.time_stamp, 1000*self.acc_x, '-r', label='acc_x')
        plt.plot(self.old_time_stamp, 1000*self.raw_acc_y, 'og', label='raw_acc_y')
        plt.plot(self.time_stamp, 1000*self.acc_y, '-g', label='acc_y')
        plt.plot(self.old_time_stamp, 1000*self.raw_acc_z, 'ob', label='raw_acc_z')
        plt.plot(self.time_stamp, 1000*self.acc_z, '-b', label='acc_z')
        plt.xlabel("Time / sec")
        plt.ylabel("Acceleration * 1e-3 / (m^2/s) ")
        plt.title("XYZ-Axis Acc Data")
        plt.savefig("Output/plot_XYZ_ACC.jpg")
        plt.show()


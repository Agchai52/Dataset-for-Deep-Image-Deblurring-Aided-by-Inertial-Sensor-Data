import numpy as np
import cv2
import matplotlib as mpl
from backports import configparser
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

        self.exposure_low = 0.05
        self.exposure_high = 0.2

        self.angular_v_mean = 0.0  # 5 if it's for rolling shutter test
        self.angular_v_std = 0.05

        self.acceleration_mean = 0.
        self.acceleration_std = 1e-4

        self.focal_length = 50.
        self.pixel_size = 2.44 * 10 ** -6
        self.image_W, self.image_H = 1280, 720

        self.intrinsicMat = np.array([[self.focal_length/self.pixel_size, 0, self.image_W/2],
                                     [0, self.focal_length/self.pixel_size, self.image_H/2],
                                     [0, 0, 1]])

        # For error data
        self.time_delay_mean = 0.03
        self.time_delay_std = 0.01
        self.total_samples = 220

        self.angular_v_noise_mean = 0.
        self.angular_v_noise_std = 0.1 * self.angular_v_std

        self.acceleration_noise_mean = 0.
        self.acceleration_noise_std = 0.1 * self.acceleration_std

        self.readout_mean = 15e-3
        self.readout_std = 6e-3
        self.total_sub = self.image_H

        # Parameter Dict for Error Effect
        self.paramDict = dict()
        self.paramDict["Sampling frequency"] = self.sample_freq
        self.paramDict["Number of poses"] = self.pose
        self.paramDict["Total samples of noisy IMU data"] = self.total_samples
        self.paramDict["Focal length"] = self.focal_length
        self.paramDict["Pixel size"] = self.pixel_size
        self.paramDict["Image Width"] = self.image_W
        self.paramDict["Image Height"] = self.image_H
        self.paramDict["Mean of gyro noise"] = self.angular_v_noise_mean
        self.paramDict["Standard deviation of gyro noise"] = self.angular_v_noise_std
        self.paramDict["Mean of acc noise"] = self.acceleration_noise_mean
        self.paramDict["Standard deviation of acc noise"] = self.acceleration_noise_std



    def generate_syn_IMU(self):
        """
         # Synthetic Inertial Sensor Data Generation
        :return:
        """

        # Generate Raw IMU data
        self.exposure = np.random.uniform(low=self.exposure_low, high=self.exposure_high)
        self.interval = self.exposure / self.pose
        self.samples = int(np.floor(self.exposure * self.sample_freq))

        self.gyro_x = 1e-5*np.random.normal(loc=self.angular_v_mean, scale=self.angular_v_std, size=(self.samples, ))
        self.gyro_y = 1e-5*np.random.normal(loc=self.angular_v_mean, scale=self.angular_v_std, size=(self.samples, ))
        self.gyro_z = np.random.normal(loc=self.angular_v_mean, scale=self.angular_v_std, size=(self.samples, ))

        self.acc_x = np.random.normal(loc=self.acceleration_mean, scale=self.acceleration_std, size=(self.samples, ))
        self.acc_y = np.random.normal(loc=self.acceleration_mean, scale=self.acceleration_std, size=(self.samples, ))
        self.acc_z = np.random.normal(loc=self.acceleration_mean, scale=self.acceleration_std, size=(self.samples, ))

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

        self.paramDict["Exposure time"] = self.exposure

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
        # Generate N = self.pose Synthetic Homography
        :return: syn_H: Array[self.pose][3x3] perfect synthetic homography


        H_i = K * (R + T * normal_vector ^ T) * inv(K)

        i = 1, 2, 3, ..., pose

        """
        self.generate_syn_IMU()
        self.rotations = self.compute_rotations()
        self.translations = self.compute_translations(self.rotations)
        self.rotations_array = np.array(self.rotations).reshape(self.pose+1, 9)

        # Ri3_0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])  # set all rows of column 3 to 0
        norm_v = np.array([0, 0, 1]).reshape(1, 3)  # norm vector of current plane

        K = self.intrinsicMat
        syn_H = []
        syn_extrinsic = []
        syn_extrinsic.append(np.array([[1,0,0],[0,1,0],[0,0,1]]))
        for i in range(1, self.pose+1):
            R = self.rotations[i]  # np.matmul(rotations[i], Ri3_0)
            T = np.matmul(self.translations[i], norm_v)
            E = R + T
            E = E / E[2][2]
            H = np.matmul(np.matmul(K, R+T), np.linalg.inv(K))
            H = H / H[2][2]
            syn_extrinsic.append(E)
            syn_H.append(H)

        self.syn_extrinsic = np.array(syn_extrinsic).reshape(self.pose+1, 9)
        return syn_H

    def create_syn_images(self, img, file_prefix, phase, isSave=True, isPlot=False):
        """
        # Generate a Synthetic Blurry Image
        :param img:  (H,W,3) ndarray size match with (self.image_H, self.image_W)
        :param isPlot: Bool True: Plot Sensor Data
        :return:  blurry image, sensor time, gyo, acc
        """
        syn_H = self.syn_homography()
        frames = []

        im_src = img
        for i in range(self.pose):
            h_mat = syn_H[i]
            im_dst = cv2.warpPerspective(im_src, h_mat, (self.image_W, self.image_H))
            frames.append(im_dst)


        frames = np.array(frames)
        blur_img = np.mean(frames, axis=0)
        self.gyro = np.stack([self.gyro_x, self.gyro_y, self.gyro_z], axis=0)
        self.acc = np.stack([self.acc_x, self.acc_y, self.acc_z], axis=0)

        error_blur_img, shift_blurry, error_gyro, error_acc, shift_time_stamp = self.add_error2data(img, syn_H, self.gyro, self.acc)

        if isSave:
            self.save_data(img, blur_img, error_blur_img, file_prefix, phase)

        if isPlot:
            self.plot_image_IMU(img, blur_img, shift_blurry, error_blur_img)
        return blur_img

    def save_data(self, reference_img, original_blur, error_blur, file_prefix, phase):
        if phase == "single":
            name_folder = "Output/"
        else:
            name_folder = "Dataset/" + phase + "/"
        name_reference = name_folder + file_prefix + "_ref.png"
        name_IMU_original = name_folder+ file_prefix + "_IMU_ori.txt"
        name_blur_original = name_folder + file_prefix + "_blur_ori.png"
        name_IMU_error = name_folder + file_prefix + "_IMU_err.txt"
        name_blur_error = name_folder + file_prefix + "_blur_err.png"
        name_param_error = name_folder + file_prefix + "_param.txt"
        name_param_config = name_folder + file_prefix + "_config.ini"
        # Output reference sharp image
        cv2.imwrite(name_reference, reference_img)

        # Original Perfect IMU data and corresponding blurry images.
        f_original = open(name_IMU_original, "w+")
        f_original.write("Timestamp Gyro_x Gyro_y Gyro_z Acc_x Acc_y Acc_z \r\n")

        for i in range(self.pose+1):
            # Timestamp  gyro_x gyro_y gyro_z acc_x acc_y acc_z
            f_original.write("%.18e %.18e %.18e %.18e %.18e %.18e %.18e \r\n" %
                             (self.time_stamp[i],
                             self.gyro_x[i], self.gyro_y[i], self.gyro_z[i],
                             self.acc_x[i], self.acc_y[i], self.acc_z[i]))

        f_original.close()
        cv2.imwrite(name_blur_original, original_blur)

        # Error IMU data and corresponding blurry images.
        f_error = open(name_IMU_error, "w+")
        f_error.write("Timestamp Gyro_x Gyro_y Gyro_z Acc_x Acc_y Acc_z\r\n")

        for i in range(self.total_samples):
            # Timestamp  gyro_x gyro_y gyro_z acc_x acc_y acc_z
            f_error.write("%.18e %.18e %.18e %.18e %.18e %.18e %.18e \r\n" %
                             (self.delay_time_stamp[i],
                              self.error_gyro[0][i], self.error_gyro[1][i], self.error_gyro[2][i],
                              self.error_acc[0][i], self.error_acc[1][i], self.error_acc[2][i]))

        f_error.close()
        cv2.imwrite(name_blur_error, error_blur)

        # Output Parameters of error data
        config = configparser.ConfigParser()
        f_error = open(name_param_error, "w+")
        config['param'] = {}

        for k, v in self.paramDict.items():
            # Timestamp  gyro_x gyro_y gyro_z acc_x acc_y acc_z
            f_error.write("%s: %f \r\n" % (k, v))
            config['param'][k] = str(v)

        f_error.close()

        with open(name_param_config, 'w') as configfile:
            config.write(configfile)

    def add_error2data(self, img,  syn_H, gyro, acc):
        """
        Add errors to perfect synthetic data
        :param img:
        :param syn_H:
        :param gyro:
        :param acc:
        :return:
        """
        # Add time delay
        shift_gyro, shift_acc, shift_time_stamp = self.add_time_delay(gyro, acc)

        # Add noise to inertial sensor data
        error_gyro, error_acc = self.add_noise2IMU(shift_gyro, shift_acc)

        # Change rotation center
        shift_blurry = self.add_rotation_center(img, syn_H)

        # # Add rolling shutter effect
        rolling_blurry = self.add_rolling_shutter(img, shift_blurry)

        # Add noise to blurry image
        error_blur_img = self.add_noise2Blurry(rolling_blurry)

        return error_blur_img, shift_blurry, shift_time_stamp, error_gyro, error_acc

    def add_time_delay(self, gyro, acc):
        """
        Add time delay to self.time_stamp, gyro and acc

        Total samples = self.total_samples (default 220)

        Original samples = self.pose + 1


        :param gyro: Array[3][self.pose+1] perfect synthetic gyro
        :param acc: Array[3][self.pose+1] perfect synthetic acc
        :return:  shifted data from start time to delayed time
        """
        time_delay = np.clip(np.random.normal(loc=self.time_delay_mean, scale=self.time_delay_std),
                             a_min=0., a_max=0.069)
        delay_index = int(np.floor(time_delay/self.interval))

        # Time
        shift_time_stamp = np.zeros(shape=self.total_samples, dtype=float)
        for i in range(self.total_samples):
            shift_time_stamp[i] = self.interval * i

        # Gyro
        shift_gyro = np.zeros(shape=(3, self.total_samples), dtype=float)
        shift_gyro[0][delay_index:delay_index+self.pose+1] = gyro[0]
        shift_gyro[1][delay_index:delay_index+self.pose+1] = gyro[1]
        shift_gyro[2][delay_index:delay_index+self.pose+1] = gyro[2]

        # Acc
        shift_acc = np.zeros(shape=(3, self.total_samples), dtype=float)
        shift_acc[0][delay_index:delay_index+self.pose+1] = acc[0]
        shift_acc[1][delay_index:delay_index+self.pose+1] = acc[1]
        shift_acc[2][delay_index:delay_index+self.pose+1] = acc[2]

        self.delay_gyro = np.array(shift_gyro).reshape((3, self.total_samples))
        self.delay_acc = np.array(shift_acc).reshape((3, self.total_samples))
        self.delay_time_stamp = shift_time_stamp
        self.paramDict["Time delay"] = time_delay
        return shift_gyro, shift_acc, shift_time_stamp

    def add_noise2IMU(self, gyro, acc):
        """
        Add noise to perfect gyro and acc
        :param gyro: Array[3][self.pose+1] perfect synthetic gyro
        :param acc: Array[3][self.pose+1] perfect synthetic acc
        :return: error_gyro and error_acc with the same size
        """
        # Gyro
        error_gyro = gyro
        add_error_gyro_0 = 1e-5*np.random.normal(loc=self.angular_v_noise_mean, scale=self.angular_v_noise_std,
                                                 size=(self.total_samples, ))
        add_error_gyro_1 = 1e-5*np.random.normal(loc=self.angular_v_noise_mean, scale=self.angular_v_noise_std,
                                                 size=(self.total_samples, ))
        add_error_gyro_2 = np.random.normal(loc=self.angular_v_noise_mean, scale=self.angular_v_noise_std,
                                            size=(self.total_samples, ))
        error_gyro[0] += add_error_gyro_0
        error_gyro[1] += add_error_gyro_1
        error_gyro[2] += add_error_gyro_2

        # Acc
        error_acc = acc
        add_error_acc_0 = np.random.normal(loc=self.acceleration_noise_mean, scale=self.acceleration_noise_std,
                                           size=(self.total_samples, ))
        add_error_acc_1 = np.random.normal(loc=self.acceleration_noise_mean, scale=self.acceleration_noise_std,
                                           size=(self.total_samples, ))
        add_error_acc_2 = np.random.normal(loc=self.acceleration_noise_mean, scale=self.acceleration_noise_std,
                                           size=(self.total_samples, ))
        error_acc[0] += add_error_acc_0
        error_acc[1] += add_error_acc_1
        error_acc[2] += add_error_acc_2

        self.error_gyro = np.array(error_gyro).reshape((3, self.total_samples))
        self.error_acc = np.array(error_acc).reshape((3, self.total_samples))

        return error_gyro, error_acc

    def add_rotation_center(self, img, syn_H):
        """
        Shift rotation center of the blurry image (average frames of poses)

        H_ = K_ * (inv(K) * H * K) * inv(K_)

        K : original intrinsic mat
        K_: new intrinsic mat, shifted rotation center
        :param syn_H: Array[self.pose][3x3] perfect synthetic homography
        :return: new blury image with shifted rotation center
        """
        rotation_o_x = np.random.normal(loc=0, scale=self.image_W / 4)
        rotation_o_y = np.random.normal(loc=0, scale=self.image_H / 4)

        self.new_intrinsicMat = np.array([[self.focal_length/self.pixel_size, 0, self.image_W/2 + rotation_o_x],
                                     [0, self.focal_length/self.pixel_size, self.image_H/2 + rotation_o_y],
                                     [0, 0, 1]])

        frames = []
        im_src = img
        K  = self.intrinsicMat
        K_ = self.new_intrinsicMat
        for i in range(self.pose):
            h_mat_ = syn_H[i]
            h_temp = np.matmul(np.matmul(np.linalg.inv(K), h_mat_), K)
            h_mat = np.matmul(np.matmul(K_, h_temp), np.linalg.inv(K_))
            h_mat = h_mat / h_mat[2][2]
            im_dst = cv2.warpPerspective(im_src, h_mat, (self.image_W, self.image_H))
            frames.append(im_dst)
            im_src = im_dst

        frames = np.array(frames)
        shift_blurry = np.mean(frames, axis=0)

        self.paramDict["Rotation center o_x"] = rotation_o_x
        self.paramDict["Rotation center o_y"] = rotation_o_y
        return shift_blurry

    def interp_rot(self, t):
        """
        :param t: current timestamp: Float
        :param acc: acc of a specific axis: Array([float])
        :return: nearest acc: Float
        """
        h_array = self.syn_extrinsic
        if t >= self.time_stamp[-1]:
            H_last = h_array[-1, :]
            return H_last.reshape((3,3))

        rot_t = np.array([0.]*9)
        for i in range(9):
            rot_t[i] = np.interp(t, self.time_stamp, h_array[:, i])

        return rot_t.reshape((3, 3))


    def add_rolling_shutter(self, img, img_blur):
        """
        Add rolling shutter effect to blurry image
        :param img_blur:
        :return: img_blur_rs

        t_y = t_readout * y / img_H
        R_new = R_last * R(t_y)
        W_new = K_ * R_new * inv(K_)
        img_blur_i = img_blur[y:y+piece_H+1]
        new_piece_i= Wrap(W_new, img_blur_i)
        """
        t_readout = np.random.normal(loc=self.readout_mean, scale=self.readout_std)
        piece_H = int(self.image_H/self.total_sub)
        H_last = self.syn_extrinsic[-1,:]
        H_last = H_last.reshape((3,3))
        K_ = self.new_intrinsicMat

        new_pieces = []
        y = piece_H  # y-1 is the row index

        while y <= self.image_H:
            # time and approximated rotation for y th row
            t_y = t_readout * y / self.image_H
            H_y = self.interp_rot(t_y)
            H_new = np.matmul(H_y, np.linalg.inv(H_last))
            W_y = np.matmul(np.matmul(K_, H_new), np.linalg.inv(K_))

            old_piece = img_blur[y-piece_H:y, :, :]
            new_piece = cv2.warpPerspective(old_piece, W_y, (self.image_W, piece_H), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS, borderMode=cv2.BORDER_REPLICATE)
            new_pieces.append(new_piece)
            y += piece_H

        img_blur_rs = np.concatenate(np.array(new_pieces), axis=0)

        self.paramDict["Readout time"] = t_readout
        return img_blur_rs



    def add_noise2Blurry(self, shift_blurry):
        """
        Add noise to a blurry image
        :param shift_blurry: (3, H, W)
        :return: error_blur_img with the same size
        """
        std_r = np.random.uniform(low=0.05, high=0.1) * 255.0 / np.sqrt(self.pose)
        noise_img = np.random.normal(loc=0, scale=std_r, size=shift_blurry.shape)
        error_blur_img = shift_blurry + noise_img

        self.paramDict["Standard deviation of noise added to the blurry image"] = std_r

        return error_blur_img

    def plot_image_IMU(self, img, blur_img, shift_blur, error_blur):

        plt.figure()
        ax1 = plt.subplot(311)
        ax1.plot(self.old_time_stamp, 100000*self.raw_gyro_x, 'or', markersize=2, label='raw_gyro_x')
        ax1.plot(self.time_stamp, 100000*self.gyro_x, '--r', label='gyro_x')
        ax1.plot(self.delay_time_stamp, 100000 * self.delay_gyro[0], '-r', label='delay_gyro_x')
        ax1.plot(self.delay_time_stamp, 100000 * self.error_gyro[0], ':r', label='error_gyro_x')
        ax1.set_xlim(0., 0.2)
        plt.ylabel("wx * 1e-5")
        plt.title("XYZ-Axis Gyro Data (rad/s)")

        ax2 = plt.subplot(312)
        ax2.set_xlim(0., 0.2)
        ax2.plot(self.old_time_stamp, 100000*self.raw_gyro_y, 'og', markersize=2, label='raw_gyro_y')
        ax2.plot(self.time_stamp, 100000*self.gyro_y, '--g', label='gyro_y')
        ax2.plot(self.delay_time_stamp, 100000 * self.delay_gyro[1], '-g', label='delay_gyro_y')
        ax2.plot(self.delay_time_stamp, 100000 * self.error_gyro[1], ':g', label='error_gyro_y')
        plt.ylabel("wy * 1e-5")

        ax3 = plt.subplot(313)
        ax3.set_xlim(0., 0.2)
        ax3.plot(self.old_time_stamp, self.raw_gyro_z, 'ob', markersize=2, label='raw_gyro_z')
        ax3.plot(self.time_stamp, self.gyro_z, '--b', label='gyro_z')
        ax3.plot(self.delay_time_stamp, self.delay_gyro[2], '-b', label='delay_gyro_z')
        ax3.plot(self.delay_time_stamp,  self.error_gyro[2], ':b', label='error_gyro_z')
        plt.ylabel("wz")

        plt.xlabel("Time / sec")
        plt.savefig("Output/plot_XYZ_Gyro.jpg")

        plt.figure()
        ax1 = plt.subplot(311)
        ax1.set_xlim(0., 0.2)
        ax1.plot(self.old_time_stamp, 1000*self.raw_acc_x, 'or', markersize=2, label='raw_acc_x')
        ax1.plot(self.time_stamp, 1000*self.acc_x, '--r', label='acc_x')
        ax1.plot(self.delay_time_stamp, 1000 * self.delay_acc[0], '-r', label='delay_acc_x')
        ax1.plot(self.delay_time_stamp, 1000 * self.error_acc[0], ':r', label='error_acc_x')
        plt.ylabel("ax * 1e-3")
        plt.title("XYZ-Axis Acc Data (m^2/s)")

        ax2 = plt.subplot(312)
        ax2.set_xlim(0., 0.2)
        ax2.plot(self.old_time_stamp, 1000*self.raw_acc_y, 'og', markersize=2, label='raw_acc_y')
        ax2.plot(self.time_stamp, 1000*self.acc_y, '--g', label='acc_y')
        ax2.plot(self.delay_time_stamp, 1000 * self.delay_acc[1], '-g', label='delay_acc_y')
        ax2.plot(self.delay_time_stamp, 1000 * self.error_acc[1], ':g', label='error_acc_y')
        plt.ylabel("ay * 1e-3")

        ax3 = plt.subplot(313)
        ax3.set_xlim(0., 0.2)
        ax3.plot(self.old_time_stamp, 1000*self.raw_acc_z, 'ob', markersize=2, label='raw_acc_z')
        ax3.plot(self.time_stamp, 1000*self.acc_z, '--b', label='acc_z')
        ax3.plot(self.delay_time_stamp, 1000 * self.delay_acc[2], '-b', label='delay_acc_z')
        ax3.plot(self.delay_time_stamp, 1000 * self.error_acc[2], ':b', label='error_acc_z')
        plt.ylabel("az * 1e-3")
        plt.xlabel("Time / sec")
        plt.savefig("Output/plot_XYZ_ACC.jpg")

        plt.figure()
        ax1 = plt.subplot(311)
        ax1.plot(self.old_time_stamp, 100000*self.raw_gyro_x, 'or', markersize=2, label='raw_gyro_x')
        ax1.plot(self.time_stamp, 100000*self.gyro_x, '--r', label='gyro_x')
        ax1.set_xlim(0., 0.5)
        plt.ylabel("wx * 1e-5")
        plt.title("XYZ-Axis Gyro Data (rad/s)")

        ax2 = plt.subplot(312)
        ax2.set_xlim(0., 0.5)
        ax2.plot(self.old_time_stamp, 100000*self.raw_gyro_y, 'og', markersize=2, label='raw_gyro_y')
        ax2.plot(self.time_stamp, 100000*self.gyro_y, '--g', label='gyro_y')
        plt.ylabel("wy * 1e-5")

        ax3 = plt.subplot(313)
        ax3.set_xlim(0., 0.5)
        ax3.plot(self.old_time_stamp, self.raw_gyro_z, 'ob', markersize=2, label='raw_gyro_z')
        ax3.plot(self.time_stamp, self.gyro_z, '--b', label='gyro_z')
        plt.ylabel("wz")

        plt.xlabel("Time / sec")
        plt.savefig("Output/plot_initial_XYZ_Gyro.jpg")

        plt.figure()
        ax1 = plt.subplot(311)
        ax1.plot(self.delay_time_stamp, 100000 * self.error_gyro[0], ':r', label='error_gyro_x')
        ax1.set_xlim(0., 0.5)
        plt.ylabel("wx * 1e-5")
        plt.title("XYZ-Axis Gyro Data (rad/s)")

        ax2 = plt.subplot(312)
        ax2.set_xlim(0., 0.5)
        ax2.plot(self.delay_time_stamp, 100000 * self.error_gyro[1], ':g', label='error_gyro_y')
        plt.ylabel("wy * 1e-5")

        ax3 = plt.subplot(313)
        ax3.set_xlim(0., 0.5)
        ax3.plot(self.delay_time_stamp, self.error_gyro[2], ':b', label='error_gyro_z')
        plt.ylabel("wz")

        plt.xlabel("Time / sec")
        plt.savefig("Output/plot_error_XYZ_Gyro.jpg")

        plt.figure()
        ax1 = plt.subplot(311)
        ax1.set_xlim(0., 0.5)
        ax1.plot(self.old_time_stamp, 1000*self.raw_acc_x, 'or', markersize=2, label='raw_acc_x')
        ax1.plot(self.time_stamp, 1000*self.acc_x, '--r', label='acc_x')
        plt.ylabel("ax * 1e-3")
        plt.title("XYZ-Axis Acc Data (m^2/s)")

        ax2 = plt.subplot(312)
        ax2.set_xlim(0., 0.5)
        ax2.plot(self.old_time_stamp, 1000*self.raw_acc_y, 'og', markersize=2, label='raw_acc_y')
        ax2.plot(self.time_stamp, 1000*self.acc_y, '--g', label='acc_y')
        plt.ylabel("ay * 1e-3")

        ax3 = plt.subplot(313)
        ax3.set_xlim(0., 0.5)
        ax3.plot(self.old_time_stamp, 1000*self.raw_acc_z, 'ob', markersize=2, label='raw_acc_z')
        ax3.plot(self.time_stamp, 1000*self.acc_z, '--b', label='acc_z')
        plt.ylabel("az * 1e-3")
        plt.xlabel("Time / sec")
        plt.savefig("Output/plot_initial_XYZ_ACC.jpg")

        plt.figure()
        ax1 = plt.subplot(311)
        ax1.set_xlim(0., 0.5)
        ax1.plot(self.delay_time_stamp, 1000 * self.error_acc[0], ':r', label='error_acc_x')
        plt.ylabel("ax * 1e-3")
        plt.title("XYZ-Axis Acc Data (m^2/s)")

        ax2 = plt.subplot(312)
        ax2.set_xlim(0., 0.5)
        ax2.plot(self.delay_time_stamp, 1000 * self.error_acc[1], ':g', label='error_acc_y')
        plt.ylabel("ay * 1e-3")

        ax3 = plt.subplot(313)
        ax3.set_xlim(0., 0.5)
        ax3.plot(self.delay_time_stamp, 1000 * self.error_acc[2], ':b', label='error_acc_z')
        plt.ylabel("az * 1e-3")
        plt.xlabel("Time / sec")
        plt.savefig("Output/plot_error_XYZ_ACC.jpg")
        plt.show()

        cv2.imshow('Reference', img / 255.0)
        cv2.imshow('Blurry_original', blur_img / 255.0)
        cv2.imshow('Blurry_center_shift', shift_blur / 255.0)
        cv2.imshow('Blurry_noise', error_blur / 255.0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

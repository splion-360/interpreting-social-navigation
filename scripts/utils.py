import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rc
from torch.utils.data import Dataset, DataLoader



class Visualiser:
    def __init__(self, frame_width, frame_height, file_dir, colors = ('lawngreen', 'skyblue', 'tomato'), 
                 tcolors = ('green', 'blue', 'red'), animation = True, fps = 5):
        '''
        Class to visualize the mouse trajectories with animation enabled. 
        
        '''

        self.FRAME_WIDTH_TOP = frame_width
        self.FRAME_HEIGHT_TOP = frame_height
        self.M1_COLOR, self.M2_COLOR, self.M3_COLOR = colors
        self.M1_TCOLOR, self.M2_TCOLOR, self.M3_TCOLOR = tcolors

        self.PLOT_MOUSE_START_END = [(0, 1), (1, 3), (3, 2), (2, 0),        # head
                                     (3, 6), (6, 9),                        # midline
                                     (9, 10), (10, 11),                     # tail
                                     (4, 5), (5, 8), (8, 9), (9, 7), (7, 4) # legs
                                    ]
        self.file_dir = file_dir
        self.fps = fps

        if animation: 
            rc('animation', html='jshtml')
        
    def set_figax(self):
        fig = plt.figure(figsize=(8, 8))
        img = np.ones((self.FRAME_HEIGHT_TOP, self.FRAME_WIDTH_TOP, 3))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        return fig, ax
    
    def plot_mouse(self, ax, pose, color):
        # Draw each keypoint
        for j in range(10):
            ax.plot(pose[j, 0], pose[j, 1], 'o', color = color, markersize=3)

        # Draw a line for each point pair to form the shape of the mouse

        for pair in self.PLOT_MOUSE_START_END:
            line_to_plot = pose[pair, :]
            ax.plot(line_to_plot[:, 0], line_to_plot[
                    :, 1], color=color, linewidth=1)
    
    def track(self, ax, pose, color, kp = 6):
        frame = pose[:,kp, :]
        ax.plot(frame[:, 0], frame[:, 1], color = color, linewidth = 2)


    def animate_pose_sequence(self, seq, start_frame = 0, stop_frame = 100, skip = 0):
        '''
        Returns the animation of the keypoint sequence between start frame
        and stop frame. Optionally can display annotations.
        '''

        image_list = []

        counter = 0
        if skip:
            anim_range = range(start_frame, stop_frame, skip)
        else:
            anim_range = range(start_frame, stop_frame)

        for j in anim_range:
            if counter % 20 == 0:
                print("Processing frame ", j)
            fig, ax = self.set_figax()
            self.plot_mouse(ax, seq[j, 0, :, :], color = self.M1_COLOR)
            self.plot_mouse(ax, seq[j, 1, :, :], color = self.M2_COLOR)
            self.plot_mouse(ax, seq[j, 2, :, :], color = self.M3_COLOR)

            self.track(ax, seq[:j, 0, :, :], color = self.M1_TCOLOR)
            self.track(ax, seq[:j, 1, :, :], color = self.M2_TCOLOR)
            self.track(ax, seq[:j, 2, :, :], color = self.M3_TCOLOR)

            ax.axis('off')
            fig.tight_layout(pad=0)
            ax.margins(0)

            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(),
                                            dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(
                fig.canvas.get_width_height()[::-1] + (3,))

            image_list.append(image_from_plot)

            plt.close()
            counter = counter + 1

        # Plot animation.
        fig = plt.figure(figsize=(8,8))
        plt.axis('off')
        im = plt.imshow(image_list[0])

        def animate(k):
            im.set_array(image_list[k])
            return im,
        ani = animation.FuncAnimation(fig, animate, frames=len(image_list), blit=True)
        return ani
    
    def fill_holes(self, data):
        '''
        Function to handle empty frames / discontinuities while processing sequences
        
        '''
        clean_data = data.copy()
        for m in range(3):
            holes = np.where(clean_data[0,m,:,0] == -1)
            if not holes:
                continue
            for h in holes[0]:
                sub = np.where(clean_data[:,m,h,0] != -1)
                if(sub and sub[0].size > 0):
                    clean_data[0,m,h,:] = clean_data[sub[0][0],m,h,:]
                else:
                    return np.empty((0))

        for fr in range(1,np.shape(clean_data)[0]):
            for m in range(3):
                holes = np.where(clean_data[fr,m,:,0] == -1)
                if not holes:
                    continue
                for h in holes[0]:
                    clean_data[fr,m,h,:] = clean_data[fr-1,m,h,:]
        return clean_data
    
    def animate(self, data:np.ndarray, fname, skip_frames = 0): 
        fpath = os.path.join(self.file_dir, fname)
        start_frame, stop_frame = 0, len(data) - 1
        clean_data = self.fill_holes(data)
        ani = self.animate_pose_sequence(
                                         clean_data,
                                         start_frame = start_frame,
                                         stop_frame = stop_frame,
                                         skip = skip_frames)
        
        ani.save(fpath + '.mp4',writer = 'ffmpeg',fps = self.fps)
        print(f"Data saved to {fpath}")


class MouseMotionData(Visualiser): 
    def __init__(self, fpath, batch_size = 50, seq_length = 5, forcePreProcess = False, infer = False,\
                 window = 5, val_fraction = 0.2, frame_height = 850, frame_width = 850, rh = 2, rw = 2):
        '''
        DataLoader object to load the coordinates of the mouse triplet dataset and containerize them as torch.Tensor. For the sake of simplicity and validation, 
        only mouse behaviour when lights are switched off is studied.

        params: 
        mix : float, (ratio describing the frames with motion to total available data) 
        '''
        super().__init__(rh, rw, "../data/videos")

        self.rawData = np.load(fpath, allow_pickle = True).item()
        self.DIR = os.path.dirname(fpath)
        self.window = window

        self.infer = infer
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.val_fraction = val_fraction
        self.numMice = 3
        self.FRAME_HEIGHT, self.FRAME_WIDTH = frame_height, frame_width

        trajectories_file = os.path.join(self.DIR, "trajectories.cpkl")

        if not (os.path.exists(trajectories_file)) or forcePreProcess: 
            print("Creating pre-processed data from raw data")
            self.frame_preprocess(trajectories_file)
        
        self.load_preprocessed(trajectories_file)

        self.reset_batch_pointer(valid = True)
        self.reset_batch_pointer(valid = False)

    def reset_batch_pointer(self, valid=False):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0            
    
    def frame_preprocess(self, data_file): 
        '''
        Function that will pre-process the each sequence of the MaBE dataset
        into data with occupancy grid that can be used.

        params:
        data_file : The file into which all the pre-processed data needs to be stored
        
        '''
        
        # Training frame data
        all_frame_data = []

        # Validation frame data
        valid_frame_data = []
        frameList_data = []

        # Read the data
        self.dataset, lframes, dframes = self.getActiveMiceData(self.rawData)
        print(f'Total lit sequences where the motion is observed: {lframes}')
        print(f'Total non-lit sequences where the motion is observed: {dframes}')
        
        sequenceIDs = list(self.dataset.keys())
        
        for sId in sequenceIDs:
            data = self.dataset[sId]
            frameList = np.arange(data.shape[0])
            numFrames = len(frameList)
            
            frameList_data.append(frameList)
            
            all_frame_data.append([])
            valid_frame_data.append([])
           

            for ind, frame in enumerate(frameList):
                miceWithPos = []

                for mice in range(self.numMice):
                    current_x = data[frame, mice, :, 0]
                    current_y = data[frame, mice, :, 1]
                    miceArr = np.ones(len(current_x)) * mice
                    miceWithPos.append([miceArr, current_x, current_y])
                
                if (ind > numFrames * self.val_fraction) or (self.infer): 
                    
                    all_frame_data[sId].append(np.array(miceWithPos).transpose(0, 2, 1))
                
                else: 
                    valid_frame_data[sId].append(np.array(miceWithPos).transpose(0, 2, 1))

        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, valid_frame_data), f, protocol=2)
        f.close() 

    def getDataSlice(self, data, begin, indices): 
        start = end = indices[begin]
        seqLen = 0
        i = begin + 1

        while i < len(indices) and indices[i] == end + 1:
            seqLen += 1
            end = indices[i] 
            i += 1    

        slicedData = data[start - self.window:start + seqLen + 1 + self.window]

        return slicedData, end, i

    def fill_holes(self, data):
        clean_data = data.copy()
        for m in range(3):
            holes = np.where(clean_data[0,m,:,0] == -1)
            if not holes:
                continue
            for h in holes[0]:
                sub = np.where(clean_data[:,m,h,0] != -1)
                if(sub and sub[0].size > 0):
                    clean_data[0,m,h,:] = clean_data[sub[0][0],m,h,:]
                else:
                    return np.empty((0))

        for fr in range(1,np.shape(clean_data)[0]):
            for m in range(3):
                holes = np.where(clean_data[fr,m,:,0] == -1)
                if not holes:
                    continue
                for h in holes[0]:
                    clean_data[fr,m,h,:] = clean_data[fr-1,m,h,:]
        return clean_data


    def normalize(self, data): 
        data =  (2 * (data / self.FRAME_HEIGHT)) - 1
        return data
    
    def getActiveMiceData(self, dataset): 
        '''
        Helper function to get the active samples (where mice moves/interacts with other mice or the environment). Mice movement is maximized 
        when the lights are switched off
        '''
        sequences = dataset['sequences']
        data = {}

        counter = 0
        nlightframes, ndarkframes = 0, 0
        for sequence in sequences.keys(): 
            chaseInd = np.where(sequences[sequence].get('annotations')[0] == 1)[-1]  
            lightInd = np.where(sequences[sequence].get('annotations')[1] == 0)[-1]
            if not len(chaseInd): 
                continue 
            
            if len(lightInd): nlightframes += 1
            else: ndarkframes += 1

            start, end = chaseInd[0], -1
            seq = sequences[sequence].get('keypoints')
            start = 0

            while end != chaseInd[-1]: 
                slicedData, end, start = self.getDataSlice(seq, start, chaseInd)
                ndata = self.normalize(slicedData)
                data[counter] = self.fill_holes(ndata)
                counter += 1

        return data, nlightframes, ndarkframes
    


    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()
        # Get all the data from the pickle file
        self.train_data = raw_data[0]
        self.frameList  = raw_data[1]
        self.valid_data = raw_data[2]
        
        counter = 0
        valid_counter = 0

        # For each dataset
        for dataset in range(len(self.train_data)):
            # get the frame data for the current dataset
            all_frame_data = self.train_data[dataset]
            valid_frame_data = self.valid_data[dataset]
            print('Training data from dataset {} : {}'.format(dataset, len(all_frame_data)))
            print('Validation data from dataset {} : {}'.format(dataset, len(valid_frame_data)))

            counter += int(len(all_frame_data) / (self.seq_length))
            valid_counter += int(len(valid_frame_data) / (self.seq_length))

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)
        self.valid_num_batches = int(valid_counter/self.batch_size)

        print('Total number of training batches: {}'.format(self.num_batches * 2))
        print('Total number of validation batches: {}'.format(self.valid_num_batches))
        # On an average, we need twice the number of batches to cover the data
        # due to randomization introduced
        self.num_batches = self.num_batches * 2

    
    def next_batch(self, randomUpdate=True):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Frame data
        frame_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            
            frame_data = self.train_data[self.dataset_pointer]

            frame_ids = self.frameList[self.dataset_pointer]

            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1]
                seq_frame_ids = frame_ids[idx:idx+self.seq_length]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)
                frame_batch.append(seq_frame_ids)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.frame_pointer += np.random.randint(1, self.seq_length)
                else:
                    self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=False)

        return x_batch, y_batch, frame_batch, d

    def next_valid_batch(self, randomUpdate=True):
        '''
        Function to get the next Validation batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.valid_data[self.valid_dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.valid_frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.valid_frame_pointer += np.random.randint(1, self.seq_length)
                else:
                    self.valid_frame_pointer += self.seq_length

                d.append(self.valid_dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=True)

        return x_batch, y_batch, d
    
    
    def tick_batch_pointer(self, valid=False):
        '''
        Advance the dataset pointer
        '''
        if not valid:
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.train_data):
                self.dataset_pointer = 0
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0  

    
    def reset_batch_pointer(self, valid=False):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0

    def visualize(self, fname):
        '''
        Animate a random sequence to verify whether the constructed dataset has no discontinuities. 
        '''
        idx = np.random.randint(0, len(self.train_data))
        data = np.array(self.train_data[idx])
        self.animate(data[:,:,:,1:], fname)
    



        


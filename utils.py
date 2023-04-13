#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import csv
import sys

# Vehicle classes 0,1,2,...22
classes = np.loadtxt('classes.csv', skiprows=1, dtype=str, delimiter=',')
print(f"class, Type, Label:\n{classes}\n")

# Vehicle labels 0,1,2
labels = classes[:, 2].astype(np.uint8)
print(f"labels: {labels}")


# Creating npy file
def write_paths(path):
    files = glob('{}/*/*_image.jpg'.format(path))
    files.sort()
    print(f"# of paths in {path}: {len(files)}")

    name = '{}/{}_paths.npy'.format(path, path)
    guid_idx = []
    
    for file in files:
        guid = file.split('\\')[-2]
        idx = file.split('\\')[-1].replace('_image.jpg', '')
        guid_idx.append(f"{guid}/{idx}")

    np.save(name, guid_idx)
    print('Saved file to `{}`'.format(name))


# Creating npy file
def write_labels(path):

    # Bounding box bin files
    files = glob('{}/*/*_bbox.bin'.format(path)) # all bounding box bin files w/ full directory path
    files.sort() # sort bounding box bin files
    print(f"# of bounding boxes for {path} snapshots: {len(files)}") # each snapshot has a bounding box of one vehicle

    # Labels files
    name = '{}/trainval_labels.npy'.format(path)
    labels_list = []
    
    # For each bounding box bin file
    for file in files:
        bbox = np.fromfile(file, dtype=np.float32) # load bounding box bin file
        '''
        Each *bbox.bin file contains an array with 11 columns. 
        Each row contains information of a bounding box: 
            0,1,2. rotation vector
            3,4,5. position (centroid x, y, z)
            6,7,8. size of the bounding box (length, width, height)
            9. class id
            10. flag
        '''
        bbox = bbox.reshape([-1, 11]) # reshape bbox.bin contents
        found_valid = False
        
        # For columns in bbox.bin
        for col in bbox:
            if bool(col[-1]):
                pass
            found_valid = True # found a vehicle in the snapshot
            class_id = col[9].astype(np.uint8) # assign class ID from bbox.bin
            label = labels[class_id] # assign corresponding label
        if not found_valid:
            label = 0
        
        labels_list.append(label)

    # Save
    labels_list = np.array(labels_list, dtype=np.uint8)
    np.save(name, labels_list)
    print('Saved file to `{}`'.format(name))


# Creating npy file
def write_rotation_vectors(path):

    # Bounding box bin files
    files = glob('{}/*/*_bbox.bin'.format(path)) # all bounding box bin files w/ full directory path
    files.sort() # sort bounding box bin files
    print(f"# of rotation vectors for {path} snapshots: {len(files)}") # each snapshot has a bounding box of one vehicle

    # Rotation vectors file
    name = '{}/trainval_rotation_vectors.npy'.format(path)
    rotation_vectors = []

    # For each bounding box bin file
    for file in files:
        bbox = np.fromfile(file, dtype=np.float32) # load bounding box bin file
        bbox = bbox.reshape([-1, 11]) # reshape bbox.bin contents
        found_valid = False

        r_vectors = [] # list to store rotation vectors

        # For columns in bbox.bin
        for col in bbox:
            if bool(col[-1]):
                pass
            found_valid = True
            r_vector = col[0:3] # assign rotation vector of vehicle from bbox.bin 
            r_vectors.extend(r_vector) # append rotation vector to the current list
        if found_valid:
            rotation_vectors.append(r_vectors) # append the list to the rotation vector list

    # Convert rotation vector list to a NumPy array
    rotation_vectors = np.array(rotation_vectors, dtype=np.float32)
    rotation_vectors = rotation_vectors.reshape([-1, 3])
    
    # Save
    np.save(name, rotation_vectors)
    print('Saved file to `{}`'.format(name))


# Creating npy file
def write_centroids(path):

    # Bounding box bin files
    files = glob('{}/*/*_bbox.bin'.format(path)) # all bounding box bin files w/ full directory path
    files.sort() # sort bounding box bin files
    print(f"# of centroids sets for {path} snapshots: {len(files)}") # each snapshot has a bounding box of one vehicle

    # Centroids files
    name = '{}/trainval_centroids.npy'.format(path)
    centroids = []

    # For each bounding box bin file
    for file in files:
        bbox = np.fromfile(file, dtype=np.float32) # load bounding box bin file
        bbox = bbox.reshape([-1, 11]) # reshape bbox.bin contents
        found_valid = False

        xyz_list = [] # list to store x, y, z values

        # For columns in bbox.bin
        for col in bbox:
            if bool(col[-1]):
                pass
            found_valid = True
            xyz = col[3:6] # assign x y z centroids of vehicle from bbox.bin 
            xyz_list.extend(xyz) # append x, y, and z values to the current list
        if found_valid:
            centroids.append(xyz_list) # append the list to the centroids list

    # Convert centroids list to a NumPy array
    centroids = np.array(centroids, dtype=np.float32)
    centroids = centroids.reshape([-1, 3])
    
    # Save
    np.save(name, centroids)
    print('Saved file to `{}`'.format(name))


# Creating npy file
def write_sizes(path):

    # Bounding box bin files
    files = glob('{}/*/*_bbox.bin'.format(path)) # all bounding box bin files w/ full directory path
    files.sort() # sort bounding box bin files
    print(f"# of bounding box size sets for {path} snapshots: {len(files)}") # each snapshot has a bounding box of one vehicle

    # Sizes files
    name = '{}/trainval_sizes.npy'.format(path)
    sizes_list = []

    # For each bounding box bin file
    for file in files:
        bbox = np.fromfile(file, dtype=np.float32) # load bounding box bin file
        bbox = bbox.reshape([-1, 11]) # reshape bbox.bin contents
        found_valid = False

        sizes = [] # list to store bounding box sizes

        # For columns in bbox.bin
        for col in bbox:
            if bool(col[-1]):
                pass
            found_valid = True
            size = col[6:9] # assign bounding box sizes of vehicle from bbox.bin 
            sizes.extend(size) # append bounding box sizes to the current list
        if found_valid:
            sizes_list.append(sizes) # append the list to the bounding box sizes list

    # Convert bounding box sizes list to a NumPy array
    sizes_list = np.array(sizes_list, dtype=np.float32)
    sizes_list = sizes_list.reshape([-1, 3])
    
    # Save
    np.save(name, sizes_list)
    print('Saved file to `{}`'.format(name))


# Creating npy file
def write_clouds(path):

    # Point clouds files
    files = glob('{}/*/*_cloud.bin'.format(path))
    files.sort()
    print(f"# of point clouds sets for {path} snapshots: {len(files)}") # each snapshot has a bounding box of one vehicle
    
    name = '{}/trainval_pointclouds.npy'.format(path)
    point_clouds = np.empty(len(files), dtype=object) # preallocation & handling different sized arrays

    # For each cloud.bin file
    for i, file in enumerate(files):
        cloud = np.fromfile(file, dtype=np.float32)
        point_clouds[i] = cloud # store each cloud as a separate object in the array

    # Save
    np.save(name, point_clouds)
    print('Saved file to `{}`'.format(name))


# Creating npy file
def write_camera_matrices(path):

    # Camera matrices files
    files = glob('{}/*/*_proj.bin'.format(path))
    files.sort()
    print(f"# of camera matrices sets for {path} snapshots: {len(files)}") # each snapshot has a bounding box of one vehicle
    
    name = '{}/trainval_camera_matrices.npy'.format(path)
    camera_matrices = np.empty(len(files), dtype=object) # preallocation & handling different sized arrays

    # For each cloud.bin file
    for i, file in enumerate(files):
        proj = np.fromfile(file, dtype=np.float32)
        proj = proj.reshape([3, 4]) # reshape to 3x4 matrix
        camera_matrices[i] = proj # store each cloud as a separate object in the array

    # Save
    np.save(name, camera_matrices)
    print('Saved file to `{}`'.format(name))


######################################## LEGACY CODES ########################################

# # Writing labels.csv file
# def write_labels2(path):

#     # Bounding box bin files
#     files = glob('{}/*/*_bbox.bin'.format(path)) # all bounding box bin files w/ full directory path
#     files.sort() # sort bounding box bin files
#     print(f"# of bounding boxes for {path} snapshots: {len(files)}") # each snapshot has a bounding box of one vehicle

#     # Labels file
#     name = '{}/trainval_labels.csv'.format(path)
#     with open(name, 'w') as f:
#         writer = csv.writer(f, delimiter=',', lineterminator='\n')
#         writer.writerow(['guid/image', 'label'])

#         # For each bounding box bin file
#         for file in files:
#             guid = file.split('\\')[-2] # scenarios (e.g. 0cec3d1f-544c-4146-8632-84b1f9fe89d3, 1dac619e-4123-42e9-bf11-8df2db3facf7, etc.)
#             idx = file.split('\\')[-1].replace('_bbox.bin', '') # snapshot numbering (e.g. 0000, 0001, 0002, etc.)

#             bbox = np.fromfile(file, dtype=np.float32) # load bounding box bin file
#             '''
#             Each *bbox.bin file contains an array with 11 columns. 
#             Each row contains information of a bounding box: 
#                 0,1,2. rotation vector
#                 3,4,5. position (centroid x, y, z)
#                 6,7,8. size of the bounding box (length, width, height)
#                 9. class id
#                 10. flag
#             '''
#             bbox = bbox.reshape([-1, 11]) # reshape bbox.bin contents
#             found_valid = False
            
#             # For columns in bbox.bin
#             for col in bbox:
#                 if bool(col[-1]):
#                     pass
#                 found_valid = True # found a vehicle in the snapshot
#                 class_id = col[9].astype(np.uint8) # assign class ID from bbox.bin
#                 label = labels[class_id] # assign corresponding label
#             if not found_valid:
#                 label = 0
            
#             writer.writerow(['{}/{}'.format(guid, idx), label])
    
#     print('Saved file to `{}`'.format(name))


# # Writing centroids.csv file
# def write_centroids(path):

#     # Bounding box bin files
#     files = glob('{}/*/*_bbox.bin'.format(path)) # all bounding box bin files w/ full directory path
#     files.sort() # sort bounding box bin files
#     print(f"# of centroids sets for {path} snapshots: {len(files)}") # each snapshot has a bounding box of one vehicle

#     # Labels file
#     name = '{}/trainval_centroids.csv'.format(path)
#     with open(name, 'w') as f:
#         writer = csv.writer(f, delimiter=',', lineterminator='\n')
#         writer.writerow(['guid/image/axis', 'value'])

#         # For each bounding box bin file
#         for file in files:
#             guid = file.split('\\')[-2] # scenarios (e.g. 0cec3d1f-544c-4146-8632-84b1f9fe89d3, 1dac619e-4123-42e9-bf11-8df2db3facf7, etc.)
#             idx = file.split('\\')[-1].replace('_bbox.bin', '') # snapshot numbering (e.g. 0000, 0001, 0002, etc.)

#             bbox = np.fromfile(file, dtype=np.float32) # load bounding box bin file
#             '''
#             Each *bbox.bin file contains an array with 11 columns. 
#             Each row contains information of a bounding box: 
#                 0,1,2. rotation vector
#                 3,4,5. position (centroid x, y, z)
#                 6,7,8. size of the bounding box (length, width, height)
#                 9. class id
#                 10. flag
#             '''
#             bbox = bbox.reshape([-1, 11]) # reshape bbox.bin contents
#             found_valid = False

#             # For columns in bbox.bin
#             for col in bbox:
#                 if bool(col[-1]):
#                     pass
#                 xyz = col[3:6] # assign x y z centroids of vehicle from bbox.bin 
#                 for a, v in zip(['x', 'y', 'z'], xyz):
#                     writer.writerow(['{}/{}/{}'.format(guid, idx, a), v])
    
#     print('Saved file to `{}`'.format(name))
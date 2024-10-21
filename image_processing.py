import numpy as np
from skimage.segmentation import flood
import cv2
import ezdxf
from ezdxf import transform
import os
import re
from scipy.spatial.distance import cdist
import imutils

debug = False # Whether to display or write debug files and information

image_folder = 'Images/' # Relative path to images folder
input_folder = 'Input/' # Relative path from images folder to input folder (Folder that will be checked for new images)
output_folder = 'Output/' # Relative path from images folder to output folder (Folder that will store output DXF files)
storage_folder = 'Storage/' # Relative path from images folder to storage folder (Folder that will store processed images)

combine_output = 'no'  # Whether to combine all detected contours on the stage into one output DXF (Note that this is a default and will be overridden by user input)
rotate_to_longest = 'no'  # Whether to rotate the output DXF so the longest generated line segment is horizontal (Note that this is a default and will be overridden by user input)

stage_size_x = 21.9 # Width in inches of lightbox stage, used for scaling

"""
---------------------- Do not change variables below this point unless necessary ----------------------
"""

stage_size_y = stage_size_x

output_regex = r' \((\d+)\).\w+' # Regex for finding existing filenames, used to create unique filename - Searches for a number in parethesis followed by a file extension "([number]).[extension]"

stage_flood_tolerance = 200 # Darkest value to be considered as a "light" pixel for flood algorithm
tool_flood_tolerance = 220
initial_min_tool_area = 80000 # Number to set the minimum area in pixels of a tool
final_min_tool_area = 10000 # Fallback until this value if no outlines are found
initial_poly_approx = 4 # Maximum distance a node can be moved during contour smoothing
initial_fill_tolerance = 25 # Inverted tolerance of the fill algorithm 0-100 (default 25)
stage_fill_tolerance = 75 # Inverted tolerance for stage detection (default 75)


input_path = image_folder + input_folder # Build input folder path
if debug:
    combine_output = True
    rotate_to_longest = False
else:
    combine_output = (input('Combine tools to single output? y/n (n):\n') or combine_output)
    combine_output = combine_output.casefold() in ('true', 'y', 'yes')
    if not combine_output:
        rotate_to_longest = (input('Rotate longest line to horizontal? y/n (n):\n') or rotate_to_longest)
        rotate_to_longest = rotate_to_longest.casefold() in ('true', 'y', 'yes')


def getUnique(regex, path, filename, extension):  # Finds highest numbered file in given format and location and returns a unique filename 1 higher
    if os.path.isfile(f'{path}{filename}.{extension}'):
        regex = rf'{filename}{regex}'
        file_count = max([int(re.fullmatch(regex, filename).group(1)) for filename in os.listdir(path) if re.fullmatch(regex, filename)] or [0]) + 1
        return f'{path}{filename} ({file_count}).{extension}'
    else:
        return f'{path}{filename}.{extension}'

def contourArea(contour_path): # Function to return area in pixels of given contour
    _, _, w, h = cv2.boundingRect(contour_path)
    return w*h

def findStage(contours, hierarchy):
    external = np.nonzero(hierarchy[0][:,3] == -1)
    num_external = len(external[0])
    if num_external < 1:
        return None
    elif num_external == 1:
        return external[0][0]
    else:
        parent = [i for i in range(0, hierarchy[0].shape[0]) if i in external[0] and hierarchy[0][i,2] != -1]
        if len(parent) < 1:
            return None
        elif len(parent) == 1:
            return parent[0]
        else:
            largestArea = -1
            largestIndex = -1
            for i in range(0, hierarchy[0].shape[0]):
                currentArea = contourArea(contours[i])
                if i in parent and currentArea > largestArea:
                    largestArea = currentArea
                    largestIndex = i
            return largestIndex

def defineContours(grayscale_image, flood_tolerance): # Function to return contours given an image and tolerance
    flood_mask = grayscale_image >= flood_tolerance
    flood_image = flood_mask.astype(np.uint8) * 255
    if debug:
        cv2.imwrite(f'{output_path}mask.png', flood_image)
    contours, hierarchy = cv2.findContours(flood_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

for dirpath, dirnames, filenames in os.walk(input_path): # dirpath, dirname, filename
    if any([filename.casefold().endswith(('.png', '.jpg', 'jpeg', '.bmp')) for filename in filenames]):
        for filename in filenames:
            if not filename.casefold().endswith(('.png', '.jpg', 'jpeg', '.bmp')):
                if debug:
                    print(f"Ignoring {filename}")
                continue
            current_fill_tolerance = initial_fill_tolerance
            current_min_tool_area = initial_min_tool_area
            current_poly_approx = initial_poly_approx
            while(True):
                source_file = f'{dirpath}/{filename}'
                source_name, _, source_extension = filename.rpartition('.')

                output_path = f'{image_folder}{output_folder}{dirpath.lstrip(input_path)}/'
                storage_path = f'{image_folder}{storage_folder}{dirpath.lstrip(input_path)}/'

                output_file = getUnique(output_regex, output_path, source_name, 'dxf')
                storage_file = getUnique(output_regex, storage_path, source_name, source_extension)


                tool_image = cv2.imread(source_file)
                grayscale_image = cv2.cvtColor(tool_image, cv2.COLOR_RGB2GRAY)

                
                if debug:
                    cv2.imwrite(f'{output_path}grayscale.png', grayscale_image)
                    print(f'Reading from {source_file}, moving to {storage_file}, writing output to {output_file}')
                else:
                    os.makedirs(output_path, exist_ok=True)
                    print(f'Reading from {source_file}')
                    
                stage_contours, stage_hierarchy = defineContours(grayscale_image, stage_flood_tolerance)
                stage_index = findStage(stage_contours, stage_hierarchy)
                stage_contour = stage_contours[stage_index]
                stage_contour = cv2.approxPolyDP(stage_contour, 100, closed=True)
                if not stage_contour.shape == (4, 1, 2):
                    input(f"\x1B[31mERROR: Error while processing {source_file}.\nThe script was unable to discern the corners of the light panel. Please ensure that the entire light panel is visible within the camera preview before capturing images.\nThis image will be marked as invalid.\nPress enter to continue.\x1B[0m")
                    if not debug:
                        os.renames(source_file, f'{source_file}.invalid')
                    break
                for point in stage_contour:
                    if point[0,0] in (0, grayscale_image.shape[1]) or point[0,1] in (0, grayscale_image.shape[0]):
                        input(f"\x1B[33mWARNING: Warning while processing file {source_file}.\nThe script may be unable to discern the proper corners of the light panel. Please ensure that the entire light panel is visible within the camera preview before capturing images.\nThis image will be processed but scaling may be incorrect. \nPress enter to continue.\x1B[0m")
                stage_contour_sorted = np.float32(stage_contour[(stage_contour*[[[1, 4]]]).sum(axis=2).argsort(axis=0)].squeeze()) # Sort stage contour corners as top left, top right, bottom left, bottom right
                    # Start by weighting the Y values higher than X by a factor of four, then get the sum of each of the coordinate pairs, then sort these. Sorting the coordinates alone cannot deconflict the bottom left and top right corners, the Y modifier biases this enough to separate them
                        # 0 1
                        # 2 3
                stage_edge_pixels = int(cdist(np.expand_dims(stage_contour_sorted[0], axis=0), np.expand_dims(stage_contour_sorted[1], axis=0))[0][0]) # This is a mostly pointless exercise but the perspective transform requires a defined size so it keeps the width of the stage from the image
                new_stage_contour = np.float32([[0, 0], [stage_edge_pixels, 0], [0, stage_edge_pixels], [stage_edge_pixels, stage_edge_pixels]])
                transformation_matrix = cv2.getPerspectiveTransform(stage_contour_sorted, new_stage_contour)
                tool_image = cv2.warpPerspective(grayscale_image, transformation_matrix, (stage_edge_pixels, stage_edge_pixels))
                
                contours, hierarchy = defineContours(tool_image, tool_flood_tolerance)
                stage_index = findStage(contours, hierarchy)
                contours = [contours[i] for i in range(0, hierarchy[0].shape[0]) if hierarchy[0][i,3] == stage_index]
                if debug:
                    tool_image = cv2.cvtColor(tool_image, cv2.COLOR_GRAY2RGB)
                    cv2.drawContours(tool_image, contours, -1, (255, 0, 0), 25)
                    test_tools_small = imutils.resize(tool_image, width=500)
                    cv2.imshow('Tool Contours', test_tools_small)
                    cv2.waitKey(0)

                contours = [contour for contour in contours if contourArea(contour) > current_min_tool_area] # Remove contours below minimum_tool_area (despeckling)

                image_scaling = stage_size_x/stage_edge_pixels
                
                lines = []
                new_contours = []
                if contours:
                    for contour in contours:
                        contour = contour.tolist()
                        new_contour = []
                        while contour:
                            x, *contour = contour
                            new_contour.append(x)
                            try: 
                                contour = contour[contour.index(x)+1:]
                            except ValueError:
                                pass
                        new_contour = np.asarray(new_contour)
                        new_contour = cv2.approxPolyDP(new_contour, current_poly_approx, closed=True)
                        lines.append(cv2.fitLine(new_contour, distType=cv2.DIST_FAIR, param=0, reps=0.01, aeps=0.01))
                        new_contours.append(new_contour)

                    cv2.drawContours(tool_image, new_contours, -1, (0,118,255), cv2.FILLED)

                    if debug:
                        cv2.imwrite(f'{output_path}contours.png', tool_image)
                    if combine_output:
                        dwg = ezdxf.new('R2010')
                        dwg.header['$MEASUREMENT'] = 0
                        msp = dwg.modelspace()
                        for contour_index in range(0,len(new_contours)):
                            contour = new_contours[contour_index-1]
                            line = lines[contour_index-1]
                            dwg.layers.new(name=f'Tool {contour_index}', dxfattribs={'color': 20})
                            contour = np.squeeze(contour)
                            for line_index in range(len(contour)):
                                msp.add_line(contour[line_index-1], contour[line_index], dxfattribs={'layer': f'Tool {contour_index}', 'lineweight': 20})
                        _ = transform.translate(msp, -np.average([np.average(new_contours[i-1], 0) for i in range(0,len(new_contours))], 0)[0])
                        _ = transform.scale(msp, sx=image_scaling, sy=-image_scaling, sz=image_scaling)
                        dwg.saveas(getUnique(output_regex, output_path, source_name, 'dxf'))
                        if not debug:
                            os.renames(source_file, storage_file)
                        break
                    else:
                        for contour_index in range(0,len(new_contours)):
                            contour = new_contours[contour_index-1]
                            line = lines[contour_index-1]
                            dwg = ezdxf.new('R2010')
                            dwg.header['$MEASUREMENT'] = 0
                            msp = dwg.modelspace()
                            dwg.layers.new(name=f'Tool {contour_index}', dxfattribs={'color': 20})
                            contour = np.squeeze(contour)
                            for line_index in range(len(contour)):
                                msp.add_line(contour[line_index-1], contour[line_index], dxfattribs={'layer': 'Tool 0', 'lineweight': 20})
                            _ = transform.translate(msp, -np.average(contour, 0))
                            if rotate_to_longest:
                                diff = contour-np.roll(contour, 1, axis=0)
                                pos_max_diff = np.argmax(np.hypot(diff[..., 0], diff[..., 1]))
                                max_diff = diff[pos_max_diff, ...]
                                _ = transform.z_rotate(msp, -np.arctan(max_diff[1]/max_diff[0]))
                            _ = transform.scale(msp, sx=image_scaling, sy=-image_scaling, sz=image_scaling)
                            dwg.saveas(getUnique(output_regex, output_path, source_name, 'dxf'))
                        if not debug:
                            os.renames(source_file, storage_file)
                        break
                else:
                    if current_min_tool_area > final_min_tool_area:
                        current_min_tool_area /= 2
                        current_fill_tolerance += 1
                        current_poly_approx += 1
                        if debug:
                            print(f"No contours found in file {source_file}, retrying with minimum tool area {current_min_tool_area} pixels, fill tolerance {current_fill_tolerance}")
                    else:
                        print(f'No contours found in file {source_file}, check tolerance and despeckling settings')
                        break

os.makedirs(input_path, exist_ok=True)

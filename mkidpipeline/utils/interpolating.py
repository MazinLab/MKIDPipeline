

def interpolate_image(input_array, method='linear'):
    """
    Seth 11/13/14
    2D interpolation to smooth over missing pixels using built-in scipy methods
    :param input_array: N x M array
    :param method:
    :return: N x M interpolated image array
    """
    final_shape = np.shape(input_array)
    # data points for interp are only pixels with counts
    data_points = np.where(np.logical_or(np.isnan(input_array), input_array == 0) == False)
    data = input_array[data_points]
    # griddata expects them in this order
    data_points = np.array((data_points[0], data_points[1]), dtype=np.int).transpose()
    # should include all points as interpolation points
    interp_points = np.where(input_array != np.nan)
    interp_points = np.array((interp_points[0], interp_points[1]), dtype=np.int).transpose()

    interpolated_frame = griddata(data_points, data, interp_points, method)
    # reshape interpolated frame into original shape
    interpolated_frame = np.reshape(interpolated_frame, final_shape)

    return interpolated_frame

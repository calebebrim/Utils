import numpy as np
from scipy.signal import triang as tf


def fuzzyPos(position, size, left=True, right=True):
    position = np.array(position) if type(position) != np.ndarray else position
    
    return position if position.sum()==0 else np.max(
        list(
            map(lambda center: createFuzzySpace(len(position),center,size,left,right),
                 np.where(position == 1)[0])),axis=0)
    


def createFuzzySpace(space_sz, center, size, left=True, right=True):
    ###   Create an fuzzy Universe arround the center #####
    #
    #     - space_sz   = The size of output array
    #     - center     = The center of triangle. The value must be an integer value compreended in [0:space_sz-1].
    #     - size       = Integer value used to calculate the basis of triangle as "b = (2*size)+1"
    #     - left|right = Boolean to set the left or the right side of triangle.
    #
    #     Usage:
    #       from calebe import fuzzyUtils as fu
    #       fu.createFuzzySpace(10,5,3)
    #       > array([0.  , 0.  , 0.25, 0.5 , 0.75, 1.  , 0.75, 0.5 , 0.25, 0.  ])
    #       fu.createFuzzySpace(10,5,3,left=False)
    #       > array([0.  , 0.  , 0., 0. , 0., 1.  , 0.75, 0.5 , 0.25, 0.  ])
    #       fu.createFuzzySpace(10,5,3,right=False)
    #       > array([0.  , 0.  , 0.25, 0.5 , 0.75, 1.  , 0. , 0. , 0. , 0.  ])
    # 
    #    Visualizing:
    #       from calebe import fuzzyUtils as fu
    #       from calebe.plotUtils import plot
    #       plot(fu.createFuzzySpace(10,5,3))
    # 
    #######################################################
    
    
    
    # print('Creating FS',space_sz,center,size)
    fu = np.zeros(space_sz)
    triang = tf(1+(2*size), True)
    if(left):
        left = center-size
        add_left = -np.min([0, 0+left])
        fu[left+add_left:center+1] = triang[add_left:size +
                                            1] if add_left > 0 else triang[:size+1]
    if(right):
        right = center+size+1
        remove_right = -np.min([0, space_sz-right])
        fu[center:right-remove_right] = triang[size:-
                                               remove_right] if remove_right > 0 else triang[size:]
    return fu

#!/usr/bin/env python
# coding: utf-8

# # Muhammad Sadqain
# # PIAIC122829

# In[1]:


import numpy as np


# # The Numpy ndarray: A Multidimensional Array Object

# In[2]:


#Function 1
# ndarray is multi dimensional container for homogenous data
data = np.array([[118, 246, 8856], [39, 879, 9854]] )
data


# In[4]:


#Function 2
data * 5


# In[5]:


data.shape


# In[7]:


# ndarray data type
data.dtype


# # Creating ndarrays

# In[8]:


dataSadqain = [1, 16, 42, 34,25]

# list to ndarray
array1 = np.array(dataSadqain)

array1


# In[12]:


# nested sequence will be converted into multi dimensional array
dataSadqainpro = [[10, 20 , 30, 50],[50, 60, 70, 80]]

array2 = np.array(dataSadqainpro)

array2


# In[13]:


print( "array2 dim :" ,array2.ndim) 
print("array1 dim :",array1.ndim)


# In[14]:


print( "array2 shape :",array2.shape) 
print("array1 shape :",array1.shape)


# In[15]:


print( "array2 datatype :" ,array2.dtype) 
print("array1 datatype :",array1.dtype)


# In[20]:


# single dim zeros array
np.ones(5)


# In[21]:


# multi dim zeros array
np.ones((3,3))


# In[22]:


# single dim ones array
np.zeros(5)


# In[26]:


# multi dim ones array
np.zeros((5, 5))


# In[27]:


# np.empty creates unintialized array
np.empty((2,2))


# In[28]:


# our favorite arange
s= np.arange(5)
s


# In[29]:


# takes another array and produce ones like array of that array size
np.ones_like(s)


# In[33]:


np.zeros_like(s)


# In[34]:


# np.identity is same as np.eye which produces identity matirix
np.eye(3)


# # Data Types of ndarrays

# In[36]:


array3 = np.array([1, 2, 3], dtype=np.float64)
array4 = np.array([1, 2, 4], dtype= np.int32)


# In[37]:



array3.dtype,array4.dtype


# In[38]:


# you can also use another array's dtype attribute
int_array = np.arange(10)
print("int_array dtype: ",int_array.dtype)

float_array = np.array([1, 3, 4, 5], dtype= float)
print("float array dtype: ",float_array.dtype)

int_to_float = int_array.astype(float_array.dtype)
print("int_to_float dtype: ",int_to_float.dtype)


# # Operations between Array and Scalars

# In[40]:


array2 = np.array([[12, 21, 34],[43, 53, 65]], dtype=float)
array2


# In[41]:


# arithematic operations
1/ array2


# # Basic Indexing and Slicing

# In[42]:


# one dimesional array

arr = np.arange(14)
arr


# In[43]:


# assigning scalar value to a slice
arr[2:7] = 12
arr


# In[44]:


arr_slice = arr[3:6]
arr_slice[1] = 7868


# In[50]:


arry2d = np.array([[11, 22, 33],[44, 55, 66],[77, 88, 99]])
arry2d


# In[51]:


arry2d[2]


# In[52]:


# to access single element of multi dim arry
print(arry2d[0][2])
#or
print(arry2d[0, 2])


# # Fancy Indexing
# ### Fancy indexing is a term adopted by NumPy to describe indexing using integer arrays. Suppose we had a 8 × 4 array:

# In[57]:


arr = np.empty((8, 4))
for i in range(5):
    arr[i] = i

arr


# In[58]:


# to select out a subset of the rows in a particular order, we can simple pass a list

arr[[4, 3, 0, 6]]  #4th row, 3rd row, 0th row, 6th row


# In[59]:


# using negative indices select row from the end

arr[[-3, -5, 7]]


# # Transposing Arrays and Swapping Axes

# In[61]:


array = np.arange(25).reshape((5, 5))
array


# In[62]:


array.T


# In[63]:


# matrix computation

arr = np.random.randn(6, 3)
arr


# # Universal Functions: Fast Element-wise Array Functions

# In[64]:


s = np.arange(14)
s


# In[65]:


# unary ufunc(because it take one value)
np.sqrt(s)


# In[66]:


# unary ufunc(it taks one value)
np.exp(s)


# In[67]:


# binary ufunc
x = np.random.randn(8)
y = np.random.randn(8)

np.maximum(x, y)


# In[70]:


np.log(14), np.log10(12), np.log2(15), np.log1p(9)


# # Data Processing Using Arrays

# In[71]:


pointsadqain = np.arange(-5, 5, 0.01)  # 1k equally spaced points


# In[73]:


# the meshgrid take two 1D arrays and produces two 2D arrays
xs, ys = np.meshgrid(pointsadqain, pointsadqain)


# In[74]:


ys


# In[75]:


xs


# In[76]:


import matplotlib.pyplot as plt


# In[78]:


sid = np.sqrt(xs **2 + ys ** 2)
sid


# # Methods for Boolean Arrays

# In[82]:


arraysadqain = np.random.randn(50)
(arraysadqain > 0).sum()   
# number of positive values


# In[83]:


# any() check if whether one or more values in any array is True 
bools = arraysadqain.any()
bools


# In[85]:


# all() check if every value is True
arraysadqain.all()


# # Random Number Generation

# In[87]:


# rando number form normal ditribution
import numpy as np
normal = np.random.normal(size=(5, 5))
normal


# In[88]:


# seed -- seed the random number generator
np.random.seed(4)
np.random.randn(2, 3)


# # Sorting

# In[89]:


arr =np.random.randn(8)
arr


# In[90]:


# sorting in place
arr.sort()


# In[91]:


arr = np.array([[5, 3, 3, 4],[4, 1, 2, 4]])

arr


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





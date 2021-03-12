
import numpy as np
np.random.seed(10)  
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras import losses


# set the data
X = np.linspace(-1,1,200)
#print(X)
np.random.shuffle(X)
#print(X)
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
#print(Y)

plt.scatter(X, Y)# plot
plt.show()


#set the data for training model 0-159 
X_train = X[:180]
Y_train = Y[:180]
#set the data for test 160-199
X_test = X[180:]
Y_test = Y[180:]


#set the neutal network model 
model = Sequential()
model.add(Dense(input_dim=1, units=1))

#build the loss function 
model.compile(loss='mse', optimizer='sgd')


#train for 500 times 
print('Training -----------')
for step in range(501):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 50 == 0:
        print("After %d trainings, the cost: %f" % (step, cost))

print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)


Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()

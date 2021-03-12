# BIASinNLP
A project relating to Bias in Natural Language Processing
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "unexpected-teens",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "np.random.seed(10)  \n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense\n",
    "import matplotlib.pyplot as plt\n",
    "from keras import losses"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "hollow-alexandria",
   "metadata": {},
   "outputs": [],
   "source": [
    "# set the data\n",
    "X = np.linspace(-1,1,200)\n",
    "#print(X)\n",
    "np.random.shuffle(X)\n",
    "#print(X)\n",
    "Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))\n",
    "#print(Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "lucky-battery",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(X, Y)# plot\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "intense-mobile",
   "metadata": {},
   "outputs": [],
   "source": [
    "#set the data for training model 0-159 \n",
    "X_train = X[:180]\n",
    "Y_train = Y[:180]\n",
    "#set the data for test 160-199\n",
    "X_test = X[180:]\n",
    "Y_test = Y[180:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "dimensional-holly",
   "metadata": {},
   "outputs": [],
   "source": [
    "#set the neutal network model \n",
    "model = Sequential()\n",
    "model.add(Dense(input_dim=1, units=1))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "stone-killer",
   "metadata": {},
   "outputs": [],
   "source": [
    "#build the loss function \n",
    "model.compile(loss='mse', optimizer='sgd')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "id": "beginning-lithuania",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training -----------\n",
      "After 0 trainings, the cost: 4.968976\n",
      "After 50 trainings, the cost: 0.971609\n",
      "After 100 trainings, the cost: 0.289882\n",
      "After 150 trainings, the cost: 0.121798\n",
      "After 200 trainings, the cost: 0.059745\n",
      "After 250 trainings, the cost: 0.031166\n",
      "After 300 trainings, the cost: 0.016963\n",
      "After 350 trainings, the cost: 0.009750\n",
      "After 400 trainings, the cost: 0.006066\n",
      "After 450 trainings, the cost: 0.004182\n",
      "After 500 trainings, the cost: 0.003218\n"
     ]
    }
   ],
   "source": [
    "#train for 500 times \n",
    "print('Training -----------')\n",
    "for step in range(501):\n",
    "    cost = model.train_on_batch(X_train, Y_train)\n",
    "    if step % 50 == 0:\n",
    "        print(\"After %d trainings, the cost: %f\" % (step, cost))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "material-photograph",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Testing ------------\n",
      "1/1 [==============================] - 0s 998us/step - loss: 0.0025\n",
      "test cost: 0.002507248427718878\n",
      "Weights= [[0.5072645]] \n",
      "biases= [2.0036924]\n"
     ]
    }
   ],
   "source": [
    "print('\\nTesting ------------')\n",
    "cost = model.evaluate(X_test, Y_test, batch_size=40)\n",
    "print('test cost:', cost)\n",
    "W, b = model.layers[0].get_weights()\n",
    "print('Weights=', W, '\\nbiases=', b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "opposed-chain",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "Y_pred = model.predict(X_test)\n",
    "plt.scatter(X_test, Y_test)\n",
    "plt.plot(X_test, Y_pred)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "magnetic-scholar",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "cannot assign to literal (<ipython-input-107-25db12913c3a>, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"<ipython-input-107-25db12913c3a>\"\u001b[1;36m, line \u001b[1;32m1\u001b[0m\n\u001b[1;33m    a = 3, b = 0.33\u001b[0m\n\u001b[1;37m        ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m cannot assign to literal\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "moral-diagram",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

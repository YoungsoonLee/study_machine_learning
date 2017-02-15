##	홍콩 과기대 김성훈 교수님의 수업. <br/>[모두를 위한 머신러닝/딥러닝 강의] 공부 정리
##### - [https://hunkim.github.io/ml/](https://hunkim.github.io/ml/)
##### - Python(v3.5) 으로 코딩.

#### 00. 개요.  
___  

+ **machine learning 이란?**   

	+ **Supervised learning**  
  		ex, Image labeling, Email spam filter, Predicting exam score  
  		type  
  		+ Predicting final exam score based on time spent - **regression**  
		+ Pass/non-pass based on time spent - **binary classification**  
		+ Letter grade(A, B, C, E and F) based on time spent - **multi-label classification**  

	+ **Unsupervised learning**  

### 01. Linear regression
___  
+ **Goal: predicting**
+ Hypothesis and Cost(Loss) function  

![image](./01_Linear_Regression/img/hypo_cost.png)

+ Goal: Minimize cost  

		minimize cost(W, b)  
           W,b   

        Gradient descent algorithm  

### 02. Logistic Classification  
___  
+ **Goal: Spam Detection(Spam ot Ham), Facebook feed(show or hide), Credit Crad Fraud Detect(legitimate/fraud) ...**  

+ Hypothesis  

![image](./02_Logistic_Regression/img/lh.png)  

+ Cost function  
![image](./02_Logistic_Regression/img/cost.png)  

+ Goal: Minimize cost  
		
![image](./02_Logistic_Regression/img/mcost.png)  

		Gradient descent algorithm  

### 03. Softmax classification: Multinomial classification  -> again  
여러개의 class가 있을때 그것을 예측.  like grade
___  

+ Hypothesis  

![image](./03_softmax_regression/img/hy.png)  

+ Cost function  

![image](./03_softmax_regression/img/cost.png)  

+ Goal: Minimize cost  

### 04. MNIST  

	...  

### 05. Neural Network  

	XOR  

### 06. TensorBoard  

![image](./05_NN/img/tb5.png)  





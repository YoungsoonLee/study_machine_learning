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
+ Hypothesis  

		H(x) = Wx + b  

		* multi-variable : use Matrix  			

+ Cost(Loss) function  

		(H(x) - y) ^ 2 (ex.y is real data)  

+ Hypothesis and Cost(Loss) function  

![image](./01_Linear_Regression/img/hypo_cost.png)

+ Goal: Minimize cost  

		minimize cost(W, b)  
           W,b   

        Gradient descent algorithm
    









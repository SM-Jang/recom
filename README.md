# Python을 이용한 개인화 추천시스템

### 해당 도서를 통한 공부
1. 2_x.py - 기본적인 추천 시스템
2. 3_x.py - 협업 필터링 추천 시스템
    1) Full matrix 만들기(user, item, rating)
    2) User간 similarty matrix (cosine, corr, jarcard)
    3) User간 rating 편차 계산 (mean, bias)
    4) User간 공통 평가 수 계산 (count)
    5) CF_knn_bias_sig 계산
    6) recom_movie

3. 4_x.py - 행렬 요인화 추천 시스템(Matrix Factorization)
    1) Full matrix R을 추정하는 방법
    2) R_hat을 만들어 R과의 차이(error) 를 최소화
    3) 최소화 방법으로 Stochastic Gradient Descent(SGD) 사용
    4) R_hat = P * Q, where P: [M, K], Q: [N, K]
    5) 사용자와 아이템의 편향을 고려(b_u, b_d)
    6) Latent Factor(K)에 따라 표현의 가지수가 다양함
    -> 너무 커지면 오버피팅하기 쉬워짐
    -> 너무 작으면 표현력의 부족으로 언더피팅
    : 개인적으로 가장 중요한 파라미터로 생각함

4. 5_x.py - Scikit-Suprise를 활용
    1) load data - Read
    2) split train/test - model_selection
    3) algorithm - BaselineOnly, KNNWithMeans, SVD, SVDpp
    4) prediction - model.test()
    5) 성능: accuracy.rmse(prediction)
    (총평) 기본적인 CF, SVD(MF) 모델을 만들기 쉬움
    하지만 Deep learning 모델을 지원하지 않음
    시간이 오래 걸림 -> cuda를 지원하는지 모르겠음
    즉, 간단한 prototype을 만드는데 유용하지만 상용화 어플로는 글쎄..

5. 
x = np.array([i*np.pi/180 for i in range(60,90*k,4)])
    np.random.seed(10)  #Setting seed for reproducibility
    y = 4*x + 7 + np.random.normal(0,3,len(x))
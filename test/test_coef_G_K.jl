G = coef_G(4)

@test G == Float64[ 1  1  1  1
                   -1  1  1  1
                    1 -1  1  1
                   -1  1 -1  1]

K = coef_K(4, 3)

@test K == Float64[  1  1  1
                    -1 -1 -1
                     1  1  1
                    -1 -1 -1]

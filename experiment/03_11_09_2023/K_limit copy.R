K_limit <- function(n){
	log(n)/log(log(n))
}

K_limit(94)

curve(K_limit, from=20, to=1e4, xlab = "sample size", ylab = "Upper limit of K")

K_limit_3 <- function(n){
	log(n)/log(log(n))-3
}
uniroot(K_limit_3, lower=10, upper=1500)

K_limit_3(93)

K_limit_4 <- function(n){
	log(n)/log(log(n))-4
}
uniroot(K_limit_4, lower=100, upper=20000)

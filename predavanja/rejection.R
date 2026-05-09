f <- function(x) {
  if (0 <= x && x <= pi / 4) {
    return (sin(x))
  } else if (pi / 4 <= x && x <= 2 + pi / 4) {
    return (-4 * x + pi + 8) / (8 * sqrt(2))
  } else {
    return (0)
  }
}

i <- 1
while (i < 10000) {
  x <- runif(1) * (2 + pi / 4)
  u <- runif(1)
  
  if (u * sqrt(2 + pi / 4) <= f(x)) {
    y[i] <- x
    i <- i + 1
  }
}

hist(y, 100, main = "Metoda odbacivanja")

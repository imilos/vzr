u <- cbind(runif(100000), runif(100000))
r <- sqrt(-2 * log(u[,1]))
theta <- 2 * pi * u[,2]
x <- r * cos(theta)
y <- r * sin(theta)

x_title <- paste("Box-Muller uzorci X\nSredina = ", format(round(mean(x), 2), nsmall = 2), "\nVarijansa = ", format(round(var(x), 2), nsmall = 2))
y_title <- paste("Box-Muller uzorci Y\nSredina = ", format(round(mean(y), 2), nsmall = 2), "\nVarijansa = ", format(round(var(y), 2), nsmall = 2))

par(mfrow = c(1,2), pty = "s")
hist(x, 100, main = x_title, xlim = c(-6, 6))
hist(y, 100, main = y_title, xlim = c(-6, 6))


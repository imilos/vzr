#
# Skriptu napisao Andrej Ilic, student IV godine OAS Informatike
# Maj 2020.
#

nmax <- 7000
r <- 5

# Pogresan metod
r1 <- runif(nmax, 0, r)
phi1 <- runif(nmax, 0, 2 * pi)

# Pravi metod
r2 <- r * sqrt(runif(nmax, 0, 1))
phi2 <- runif(nmax, 0, 2 * pi)

# Konverzija u dekartove koordinate
x1 <- r1 * cos(phi1)
y1 <- r1 * sin(phi1)
x2 <- r2 * cos(phi2)
y2 <- r2 * sin(phi2)

par(mfrow = c(1,2), pty = "s")
plot(x1, y1, pch = 20, cex = 0.5, col = "Red", main = "Pogresno")
plot(x2, y2, pch = 20, cex = 0.5, col = "Blue", main = "Tacno")
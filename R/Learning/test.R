# Define constants and variables
h <- 4.136e-15   # Planck's constant (eV s)
m <- 9.11e-31    # Mass of electron (kg)
x <- seq(-1, 1, length.out = 1000) * 1e-9   # Position in nanometers

# Define the wave function and probability density functions
psi <- function(x, omega) {
  sqrt(m*omega/(pi*h))*exp(-m*omega*x^2/(2*h))
}
rho <- function(x, omega) {
  psi(x, omega)^2
}

# Plot the wave function and probability density for three different angular frequencies
curve(psi(x, 0.1), from = -1, to = 1, col = "blue", lwd = 2, xlab = "Position (nm)", ylab = "Amplitude")
curve(rho(x, 0.1), add = TRUE, col = "blue", lwd = 2)
curve(psi(x, 0.2), add = TRUE, col = "green", lwd = 2)
curve(rho(x, 0.2), add = TRUE, col = "green", lwd = 2)
curve(psi(x, 0.3), add = TRUE, col = "red", lwd = 2)
curve(rho(x, 0.3), add = TRUE, col = "red", lwd = 2)
legend("topright", legend = c("0.1 eV", "0.2 eV", "0.3 eV"), col = c("blue", "green", "red"), lwd = 2)

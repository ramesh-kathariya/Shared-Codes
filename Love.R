Love <- data.frame(t=seq(0,2*pi,by=0.1))
xhrt <- function(t) 16 * sin(t)^3
yhrt <- function(t) 13*cos(t)-5*cos(2*t)-2*cos(3*t)-cos(4*t)
Love$y = yhrt(Love$t)
Love$x = xhrt(Love$t)

with (Love, plot(x, y, col = "red", cex = 0))
with(Love, polygon(x, y, 
                   col = "red"))


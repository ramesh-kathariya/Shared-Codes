rm(list = ls())
# Title: "Basic of R"
# Author: "Ramesh Kathariya"
# Email: "rameshkathariya9@gmail.com"


A <- c(1, 2, 3, 4, 5, 5, 5, 6, 7, 8);
y <- table(A) # Conterting the sets into frequency table
y


names(y)[which(y == max(y))] # Mode

median(A)                   # Median

mean(A)                     # Mean

range(A)                   # Range

min(A)                    # Minimum value
max(A)                    # Max Value

res <- range(A);
diff(res)                 # Diff between max & min value

IQR(A)                   # Inter-quartile Range

quantile(A)              # Quartiles

quantile(A, 0.25)        # Q1 Quartile
quantile(A, 0.75)        # Q3 Quartile

 # Sometimes we have to create our own fuctions, here is the method of creating the function
SE <- function(x){sd(x)/sqrt(length(x))}    # Standard Error
SD <- function(x){sqrt(var(x))}             # Standard Deviation
#################### na.rm = TRUE

N <- length(A)         # Count/Length of A

pop_var <- var(A) * (N -1)/N # Population Variance
sam_var <- var(A)
pop_var; sam_var          # Print value of variances

sqrt(pop_var)           # Pop SD
sd(A)                   # Sample SD

############ Normal Distribution ##############
 #We can check it using hsitogram
hist(A)              # hist(data, break = 15)
qqnorm(A)            # qqnorm(data$x)       # To see whether data is normally distributed
qqline(A)            # qqline(data$x)       # To see the distribution of the data
  # In the Q-Q plot, if the points do not deviate away
       #from the line, the data is normally distributed.
boxplot(A)          # To see the distribution of data 
shapiro.test(A)     # If p>0.5, data doesn't deviate from normal distribution

seed <- set.seed(123)
b  <- rnorm(50, 3, 0.5) # Generating random number/deviates from random distribution with n = 50, mean = 3 & sd = 0.5
                        # for default mean = 0 and sd = 1
hist(b, breaks = "Sturges", col = 3: 10) #instead of "Sturges", you can use numbers


pnorm(1.9, 3, 0.5) # for P(x<1.9) pnorm gives distribution function, to calculate cumulative distribution function (CDF)
1-pnorm(1.9, 3, 0.5) # for P(x<1.9), 
qnorm(0.95, 3, 0.5) #inverse CDF and to look for P-th Quantile, qnorm gives quantile function
                    # q = 0.95 gives 95-th percentile
 ### Modality of a distribution can be seen by the number of peaks when
    #we plot the histogram; Types: unimodal, bimodal and multimodal

 ### Skewness - measure of symmetry negative(left), positive (left), symmetrical
               # normal distn, heavy tailed, light tailed


########### Measuring Kurtosis and Skewness
#install.packages("moments")
library(moments)
require(moments)
skewness(b)
kurtosis(b)

######### Binomial Distribution #########
 #Bino Distn has 2 outcomes (sucess & failure) and can be thought as probability 
 # of success and failure

dbinom(32, 100, 0.5) # To get probability mass function, Pr(X = X) or P(X = 30)
pbinom(32, 100, 0.5) # To get cummulative distribution function P(X<= x)
qbinom(0.3, 100, 0.5) # To look up in 30-th quantile of binomial distribution
A <- rbinom(1000, 100, 0.5) #A <- rnorm(1000, 100, 0.5)
hist(A, breaks = 20, col = 3:9)


##########Summary and Str Function #############
summary(A)
str(A)

##############################################################
##############################################################
############# DATA VISUALIZATION #############################

# Activities : Bar chart, Histogram, Line chart, Pie chart, Scatter plot, Box plot, Scatterplot matrix
              # plotting decision tree and social network analysis graphs
           # Packages: ggplot2 and Ploty JS

data <- c(4, 6, 7, 9, 10, 20, 12, 8);
      
         # Making barplot
barplot(data, xlab="X-axis", ylab="Y-axis", 
        main="Bar Chart 1",
          col="green"); abline(h = 0)

        # Exporting Bar chart into an image file
png(file="E:/Ramesh/M.Sc III Sem/R Practice Jan 9/barchart1.png")

dev.off()            # Removing the plotted graphs


        # Making the bar graph horizontal
barplot(data, xlab="X-axis", ylab="Y-axis", 
        main="Bar Chart 1",
        col="green", horiz = TRUE); abline(v = 0)
       
         #Loading inbuilt mtcars data
data(mtcars)
data <- table(mtcars$gear, mtcars$carb)
barplot(data, xlab="x-axis", ylab="y-axis", main="bar chart
1", col=c("red", "blue", "yellow")); abline(h = 0)

           # To plot a grouped bar chart, use beside = TRUE
barplot(data, xlab="x-axis", ylab="y-axis", main="bar chart
1", col=c("red", "blue", "yellow"), beside = TRUE); abline(h = 0)

########### Plotting Histogram ############
set.seed(123)
data1 <- rnorm(100, 5, 3)
hist(data1, main="histogram", xlab="x-axis", col="green",
     border="blue")
# changing breaks , breaks=10)
hist(data1, main="histogram", xlab="x-axis", col="green",
     border="blue", breaks=10);
 
###### To add density line, use freq = False <- hist is plotted based on probability
      # we can use line() function to add density line
hist(data1, main="histogram", xlab="x-axis", col="green",
     border="blue", breaks=10, freq = FALSE);
lines(density(data1), col = "red")

 ########## Making line chart and Pie chart
x <- c(1, 2, 3, 4, 5, 6, 8, 9);
y <- c(3, 5, 4, 6, 9, 8, 2, 1);
#par(mfrow =(c(1,2)))
plot(x, y, type="l", xlab="x-axis", ylab="y-axis", main="line
graph", col="blue");

 # Adding more data plus plotting double line
x.1 <- c(2, 3, 4, 6, 7, 8, 9, 10);
y.1 <- c(6, 3, 5, 1, 5, 3, 4, 8);
plot(x, y, type="l", xlab="x-axis", ylab="y-axis", main="line
graph", col="blue");
lines(x.1, y.1, type="o", col="green");

dev.off()
 ### Creating  a pie chart using pie() function
z <- c(10, 30, 60, 10, 50);
labels <- c("A", "B", "C", "D", "E");
pie(z,labels, main = "Pie Chart")
 ### Creating 3D Pie chart
library(plotrix)
pie3D(x, labels = labels, explode = 0.1, main = "Pie Chart")

 ########## Scatter plot and Box plot ##########
plot(x, y, cex = 1, pch = 16, col = "green")


########## Creating a box plot #########
var1 <- rnorm(100, mean=3, sd=3);
var2 <- rnorm(100, mean=2, sd=2);
var3 <- rnorm(100, mean=1, sd=3);
data <- data.frame(var1, var2, var3)

boxplot(data, main = "Box plot", notch = FALSE, 
        varwidth = TRUE, col = c(3:6))
dev.off()

boxplot(data, main = "Box plot", notch = TRUE, 
        varwidth = TRUE, col = c(3:6))  # setting notch in the box plot

### ##### Scatter Plot Matrix ########
set.seed(12);
var1 <- rnorm(100, mean=1, sd=3)
var2 <- rnorm(100, mean=1, sd=3)
var3 <- rnorm(100, mean=1, sd=3)
var4 <- rnorm(100, mean=2, sd=3)
var5 <- rnorm(100, mean=2, sd=3)
data <- data.frame(var1, var2, var3, var4, var5);
par(mfrow = c(1,2))
plot(var1, col = "blue", pch = 16)
pairs(data, col = "red")

#********************************************************************************
#*

#*
#*
#loading data in csv format

biomass_e <- c(30, 50, 60, 70, 80, 75)
biomass_w <- c(120, 121, 225, 250, 230, 210)

#checking normality: Boxplot, Shapiro Test, Histogram,
boxplot(biomass_e, biomass_w) # the biomass_w is not normally distr


shapiro.test(biomass_e) #Assumption null: data is normal,
#>> Since p > 0.5, we fail to reject null hypothesis
#> that is, the data is normally distributed
shapiro.test(biomass_w)
hist(biomass_e)
hist(biomass_w) # how is normal histogram?



# Create a Q-Q plot
par(mfrow= c(1,2))
qqnorm(biomass_e)
qqline(biomass_e, col = "red")  # Add a reference line
qqnorm(biomass_w)
qqline(biomass_w, col = "red")
# Add title and labels
#title("Q-Q Plot")

# Power of Test
# Calculate power for a two-sample t-test
result <- power.t.test(n = 50, # sample size
                       delta = 0.5, # true difference in means
                       sd = 1, # standard deviation
                       sig.level = 0.05, # significance level
                       type = "two.sample", # type of t-test
                       alternative = "two.sided") # one- or two-sided test

# Print the result
print(result)

#Second assumption ## Before T Test we check whether variance of both sample are equal

var.test(biomass_e, biomass_w) # null: variance are equal
# Welch t test?

########################################################################

#             PERFORMING ONE-SAMPLE T-TEST

###########################################################################

# suppose we have hypothetical population mean  of biomass for eastern side = 50

t.test(biomass_e, mu = 50, alpha = 95)

#########################################################################

#            PERFORMING PAIRED T-TEST
###############################################################################

#  Following data is weight (gm) of mice before and after treatment. 
#      Test for the effectiveness of the treatment in increasing the body weight 
#      (assume the normality in the data).
#   Before	350	550	230	340	120	300	330	310	305
#   After 	360	557	300	350	125	307	340	320	330
Before <- c(350, 550, 230,	340,	120,	300,	330,	310,	305)
After <- c(360,	557,	300,	350,	125,	307,	340,	320,	330)
t.test(Before, After, paired = TRUE)

##########################################################################

#                  Independent t-test

#######################################################################
# you don't want to have Welch t test
t.test(biomass_e, biomass_w, alternative = "two.sided", 
       conf.level = 0.99, var.equal = TRUE)

# use alternative = "less" or "greater" for left and right sided test

# By default, it calculates Welch t-test 
#  (It is a parametric test and used when the samples have unequal variance)

t.test(biomass_e, biomass_w) # Null: mean of eastern area = mean of western area
# alternative: Null: mean of eastern area not equals to mean of western are
# above is two tailed t test

t.test(biomass_e, biomass_w, alternative = "less", 
       conf.level = 0.99, var.equal = F) # LESS represent one tailed


# ******************************************************************************

#  What if my data is not normal?

# one-sample t-test --> Wilcox Signed Rank Test

# Independent S. t-test ---> Man Whitney U Test (Wilcox Ranked Sum Test)

# One-way ANOVA ---> Kruskal Wallis Test

# ANCOVA ---------> ANCOVA of Ranks


#********************   UNQEUAL VARIANCE   *************************************
#*One-way ANOVA -----> Welch's ANOVA
#*Independent S. t-test -----------> Welchs t-test
#*
#*
#******************  Are dependent **********************************************
#*
#* ANOVA/Indep S. T-Test -----------> Bootstraping/ permutation
#


#############################################################

#                    Alternatives

#############################################################
wilcox.test(biomass_e, mu = 50, alpha = 95)     #  Wilcox Signed Rank Test

wilcox.test(Before, After, paired = TRUE,       # Wilcox Signed Rank Test
            alternative = "less")

wilcox.test(biomass_e, biomass_w)      #  Man Whitney U Test (Wilcox Ranked Sum Test)

kruskal.test(biomass_e, biomass_w)    # Kruskal Wallis Test

################################################################################

#                  ANOVA - Working with Multiple Factor

################################################################################
 #                     One-Way ANOVA

################################################
dat <- ToothGrowth
str(my_data)
mod_1 <- aov(len~dose,data= dat) #tilday sign
summary(mod_1) 
# Now we are performing Tukey HSD for evaluating which dose has most significant effect on lenght
TukeyHSD(MO)

# Error: It gives error as we need to change dose to factor
dat$dose <- as.factor(as.numeric(dat$dose))
(mod_2 <- aov(len~dose, dat))
summary(mod_2)
TukeyHSD(mod_2)

#                     Two-Way ANOVA

################################################

mod_3 <- aov(len~supp+dose,data = dat) #  Additive Model
summary(mod_3)

mod_4<- aov(len~supp*dose,data = dat)  #  Multiplicative Model
summary(mod_4)
# If interaction seems significant we use interactive model otherwise use subtrative model

# We will check the fitness of the model using AIC - Lower the AIC better the fit
AIC(mod_3, mod_4)

################################################################################

#                  ANCOVA - Integrating a covariate in ANOVA

################################################################################

car <- mtcars[,c("am","mpg","hp")] #only selected these variables
names(car)
str(car) # since all the data set is in numeric we didn't converted it to factor
mod_5 <- aov(mpg~hp+am, data = car)
summary(mod_5)

################################################################################

#                  Correlation Test

################################################################################
#creating data set
tree <- data.frame(Biomass = c(45,54,67,78,96),
                   Density = c(12,45,54,58,70))

str(tree)
plot(tree$Biomass, tree$Density, cex = 2, pch = 8, col = "red")


#to check significance
cor(tree$Biomass, tree$Density) # It measures Karl-person correlation by default
cor.test(tree$Biomass, tree$Density)

cor(tree$Biomass, tree$Density, 
    method="pearson")

cor(tree$Biomass, tree$Density, 
    method="spearman")


mod_6 <- lm(Density~Biomass, data = tree) # for linear regression
abline(mod_6, lwd=2, col="red", lty = 6)
text(55, 60, "We are practicing txt", col = "red", adj = c(0, -1))
summary(mod_6)

################################################################################

#                  Chi-Square Test

################################################################################
Deer <- c(50, 75)
chisq.test(Deer, p = c(1/3,2/3)) #  "p" is the proportion in which the populations
Axis <- c(50,75)
Muntiacus <- c(65,15)
Deer.Chitwan <- data.frame(Axis,Muntiacus)
Deer.Chitwan
row.names(Deer.Chitwan) <- c("Forest","Riverine")
chisq.test(Deer.Chitwan)


Alex_para <- c(32,65)

Rose <- c(70,85)


Bird_ktm <- data.frame(Alex_para, Rose) 

Bird_ktm

row.names(Bird_ktm)<- c("Pine","Schima")

chisq.test(Bird_ktm)

# Yates continuity correction = for small frequencies < 5, we ues correct = T
#  Alternatively we can use fisher.test()

chisq.test(Bird_ktm, correct = TRUE)

fisher.test(Bird_ktm)




################################################################################

#                  Package Reshape

################################################################################


# changing into long data format
library(reshape2)
biomass <- data.frame(biomass_e, biomass_w, 
                      code = c("a", "b", "c", "a", "b", "a")) # The data frame is in short data format
biomass

biomass1 <- melt(biomass, id.vars =  "code")
biomass1

# changing into short data format or wide data format

biomass2 <- dcast(biomass1, variable~value, mean)

################################################################################

#                   PACKAGE: SCIPLOT - Function Bargraph.CI
#                     plotting standard error bar
#*******************************************************************************

library(sciplot)

View(biomass1)
bargraph.CI(variable, value, group = code, 
            data = biomass1, 
            col =4:5) #bargraph only works on long data format
abline(h=0)

# To extend the y-axis

bargraph.CI(variable, value, data = biomass1, 
            col =4:5, ylim = c(0, 250))
abline(h=0)
abline(h=150, lwd =3, lty = 3, col = "red")
abline(v = 1.5, lwd = 3, lty =3, col = "blue")

lineplot.CI(variable, value, data = biomass1, col =4:5)


#removing data
rm(biomass)
#Adding one variable in data frame
biomass1$grazing <- c(rep("grazed", 6), rep("ungrazed", 6))
#adding another
biomass1$altitude <- c(120, 130, 140, 120, 125, 50, 29, 43, 56, 78, NA, NA)
biomass1


mean(biomass1$altitude, na.rm =TRUE)

#********************************************************************************
#*
#*

#     Working with Multivariate Tools


#Is an unconstrained ordination (indirect multivariate)
install.packages("vegan") # Installing Vegan Package
require(vegan)            # loading package "vegan"
data("varespec")          # Loading in-built data "varespec" from the package
head(varespec, n = 6)


# Perform PCA (Method 1)
#pca_result <- prcomp(iris[, 1:4], scale = TRUE)  # Select columns 1 to 4 and scale the data
pca_result <- prcomp(varespec, scale = FALSE)

# Plot the scree plot to visualize the major components of PCA
screeplot(pca_result, type = "line", npcs = 10, main = "Scree Plot")

pca_result$sdev         # Standard deviations of principal components

pca_result$rotation     # Variable loadings

pca_result$x           # Individual coordinates


biplot(pca_result, scale = 0)  # Visualizing result obtained from PCA
pca <- rda(varespec)
biplot(pca)
pairs(pca_result$x[, 1:3])  # Visualize PC1 to PC3


#install.packages("rgl")     # Installing package rgl:
# RGL provides medium to high level functions for 3D
#  interactive graphics, including functions modelled 
#    on base graphics
library(rgl)                # loading package "rgl"
plot3d(pca_result$x[, 1:3], type = "n")  # Visualize PC1 to PC3 in 3D
points3d(pca_result$x[, 1:3], col = "blue")  # Add data points


# NMDS, is an indirect gradient analysis approach that creates an ordination 
#     based on a dissimilarity or distance matrix. It attempts to represent the
#     between objects in a low-dimensional space. 
# Unlike other methods that attempt to maximize the correspondence between objects in an ordination,
# NMDS is a rank-based approach which means that the original distance data is substituted with ranks.


# dist <- vegdist(varespec) | 
#my_nmds <- metaMDS(dist)
#plot(my_nmds, cex = 0.5, type = "text") #type = "t" also works though "text" also works

nmds <- metaMDS(varespec) #metaMDS uses Bray-Curtis distance by default 
#   if you want to specify any other distance matrix 
#     you can use above process or use distance =  "euclidean" or "jaccard" or any other 
plot(nmds, cex = 0.5, type = "text") #type = "t" also works though "text" also works

dev.off()               # Remove the plots

# Performing CA
#install.packages("ca")
#library(ca)
#library(FactoMineR); library(factoextra)
#install.packages("factoMineR")
#CA <- CA(varespec, graph = F)
#summary(CA)
#plot(CA)




#             Performing CA

my_ca <- cca(varespec)
summary(my_ca)
plot(my_ca)




#              Performing DCA

dca_result <- decorana(varespec)
summary(dca_result)#: Provides a summary of the DCA results.
plot(dca_result)#: Generates a detrended correspondence analysis plot.
ordiplot(dca_result)#: Generates a species-environment biplot.


#*********************************************************************************
######### Constrained ordination ######## Direct Gradient Analysis
#**********************************************************************************

#         *Performing CCA*
#         --------------
data("varechem"); data("varespec")
View(varechem);
View(varespec)
par(mfrow = c(1,2))
my_cca1 <- cca (varespec~., varechem)
#my_cca2 <- cca (varechem~., varespec)
plot(my_cca1)
#plot(my_cca2)

dev.off()
###RDA
#png(filename = "fig.png", width = 1000,  height = 800) # To save the plot





#            *Performing  RDA*
#           ----------------
data("varechem")
my_rda <- rda(varespec~., varechem)
plot(my_rda, cex = 6)



############################## CLUSTER ANALYSIS #####################

my_dist <- vegdist(varespec, method = "euclidean") # By default it is Bray-Curtis Distance
my_clust <- hclust(my_dist)
plot(my_clust)



#####K-mean clustering

data(iris)
iris
iris_new <- iris[, c(1,2,3,4)]; head(iris_new)
iris_class <- iris[,"Species"]; head(iris_class)
result <- kmeans(iris_new, 3)
print(result$cluster)

plot(iris_new[c(1,2)], col = result$cluster)

plot(iris_new[c(3,4)],
     col = result$cluster)


iris(result$cluster, iris_class)

sep <- iris[, c(1,2)]

# Hierarchical Clustering

dist <- dist(sep)
h_clust <- hclust(dist)
plot(h_clust)



# For GLM and GLLM codes will soon be shared











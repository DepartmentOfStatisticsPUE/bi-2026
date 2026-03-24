## -----------------------------------------------------------------------------
library(survey)
library(sampling)


## -----------------------------------------------------------------------------
# Define parameters
regions <- c("North", "South", "East")
cities_per_region <- 4
individuals_per_city <- c(20, 40, 60, 80)
city_labels <- paste0(rep(regions, each = cities_per_region), 1:cities_per_region)

# Create cities data frame
cities_df <- data.frame(
  region = rep(regions, each = cities_per_region),
  city = city_labels,
  num_individuals = rep(individuals_per_city, times = 3)
)

# Create population data frame
population <- do.call(rbind, lapply(1:nrow(cities_df), function(i) {
  data.frame(
    region = cities_df$region[i],
    city = cities_df$city[i],
    individual_id = 1:cities_df$num_individuals[i]
  )
}))

# Add target variable Y (income)
mean_income <- c(North = 50000, South = 40000, East = 45000)
set.seed(123)
population$Y <- rnorm(nrow(population), mean = mean_income[population$region], sd = 10000)

# Check population size
nrow(population)  # Should be 600


## -----------------------------------------------------------------------------
set.seed(123)
sample_indices <- sample(1:nrow(population), size = 100, replace = FALSE)
srs_sample <- population[sample_indices, ]

# Declare design
design <- svydesign(ids = ~1, data = srs_sample, fpc = ~rep(600, 100))

# Estimate mean of Y
svymean(~Y, design)

cat("True Y:", mean(population$Y))


## -----------------------------------------------------------------------------
# Draw stratified sample
set.seed(123)
strat_sample <- sampling::strata(data = population, stratanames = "region", 
                                 size = c(33, 33, 33), method = "srswor")
sample_indices <- strat_sample$ID_unit
strat_sample_data <- population[sample_indices, ]

# Declare design
strat_sample_data$stratum_size <- 200  # Each region has 200 individuals
design <- svydesign(ids = ~1, strata = ~region, data = strat_sample_data, fpc = ~stratum_size)

# Estimate mean of Y
svymean(~Y, design)
cat("True Y:", mean(population$Y))


## -----------------------------------------------------------------------------
set.seed(123)
selected_cities <- sample(unique(population$city), size = 4)
cluster_sample <- population[population$city %in% selected_cities, ]

# Declare design (12 total clusters)
design <- svydesign(ids = ~city, data = cluster_sample, fpc = ~rep(12, nrow(cluster_sample)))

# Estimate mean of Y
svymean(~Y, design)


## -----------------------------------------------------------------------------
set.seed(123)
city_by_region <- tapply(population$city, population$region, unique)
selected_cities <- unlist(lapply(city_by_region, function(x) sample(x, size = 2)))
cluster_strat_sample <- population[population$city %in% selected_cities, ]

# Declare design (4 cities per stratum)
cluster_strat_sample$cluster_fpc <- 4
design <- svydesign(ids = ~city, strata = ~region, data = cluster_strat_sample, fpc = ~cluster_fpc)

# Estimate mean of Y
svymean(~Y, design)


## -----------------------------------------------------------------------------
# Draw Poisson sample
set.seed(123)
pi_i <- 100 / 600
include <- rbinom(nrow(population), size = 1, prob = pi_i)
poisson_sample <- population[include == 1, ]

# Declare design
design <- svydesign(ids = ~1, data = poisson_sample, probs = ~rep(pi_i, nrow(poisson_sample)))

# Estimate mean of Y
svymean(~Y, design)


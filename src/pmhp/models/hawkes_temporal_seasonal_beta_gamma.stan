functions {

  // ------------------------------------------------------------
  // ---------------------- Model PMHP --------------------------
  // ------------------------------------------------------------
  //
  // For the normalized exponential kernel
  //   phi_ij(u) = alpha_ij * beta_ij * exp(-beta_ij * u),
  //
  // the integrated contribution of one parent event over the
  // remaining observation window is
  //   alpha_ij * (1 - exp(-beta_ij * (T_end - t_n))).
  //
  // This part is identical for the seasonal and constant-baseline
  // specifications.
  //
  // Author: Persia Luca (2026), Università della Svizzera italiana, Lugano, Switzerland
  // Notes: partially based on https://github.com/younesszs/O2O 

  real partial_compensator(
      array[] int slice_types,
      int start,
      int end,
      vector times,
      vector marks,
      matrix alpha,
      matrix beta,
      real T_end,
      int D
  ) {

    real partial_log_lik;
    partial_log_lik = 0.0;

    for (kk in 1:size(slice_types)) {

      int i;
      int parent;

      i = start + kk - 1;
      parent = slice_types[kk];

      for (d in 1:D) {

        partial_log_lik -=
          alpha[d, parent]
          *
          marks[i]
          *
          (
            1.0
            -
            exp(
              -beta[d, parent]
              *
              (T_end - times[i])
            )
          );
      }
    }

    return partial_log_lik;
  }
}


data {

  // EVENT DATA ------------------------------------------------

  int<lower=1> N;
  int<lower=1> D;
  vector[N] times;
  array[N] int<lower=1, upper=D> types;
  vector<lower=0>[N] marks;
  real<lower=0> T_end;

  // BASELINE ---------------------------------------------------
  // use_seasonality = 1: proposed annual sine-cosine baseline
  // use_seasonality = 0: constant baseline benchmark

  int<lower=0, upper=1> use_seasonality;

  // Continuous annual seasonality inputs -----------------------
  vector[N] season_cos_event;
  vector[N] season_sin_event;
  int<lower=1> G;
  vector[G] season_cos_grid;
  vector[G] season_sin_grid;
  vector<lower=0>[G] day_exposure;

  // Prior controls ---------------------------------------------
  real<lower=0> mu_prior_shape;
  real<lower=0> mu_prior_rate;
  real<lower=0> alpha_prior_a;
  real<lower=0> alpha_prior_b;
  real<lower=0> beta_prior_shape;
  real<lower=0> beta_prior_rate;
  real<lower=0> season_sd;
}


parameters {

  // Baseline intensities --------------------------------------
  vector<lower=1e-9>[D] mu;

  // Branching-ratio matrix ------------------------------------
  matrix<lower=1e-9,upper=1 - 1e-9>[D, D] alpha;


  // Exponential decay-rate matrix -----------------------------
  matrix<lower=1e-9>[D, D] beta;

  // Seasonal Fourier coefficients -----------------------------
  // The parameter vector has dimension two for the seasonal
  // specification and dimension zero for the constant benchmark.
  // Benchmark does not estimate unnecessary seasonal parameters.

  vector[2 * use_seasonality] season_coef;
}


transformed parameters {

  vector[G] season_factor_grid;
  real season_log_norm;
  real seasonal_baseline_integral;
  real season_cos_coef;
  real season_sin_coef;

  // SEASONAL BASELINE and NORMALIZATION ----------------------

  if (use_seasonality == 1) {

    vector[G] f_grid;
    season_cos_coef = season_coef[1];
    season_sin_coef = season_coef[2];

    for (g in 1:G) {
      f_grid[g] = season_cos_coef * season_cos_grid[g] + season_sin_coef * season_sin_grid[g];
    } 

    season_log_norm = log_sum_exp(f_grid) - log(1.0 * G);
    for (g in 1:G) {
      season_factor_grid[g] = exp(f_grid[g] - season_log_norm);
    }
    seasonal_baseline_integral = dot_product(day_exposure, season_factor_grid);
  }


  // CONSTANT BASELINE -------------------------------------------

  else {
    season_cos_coef = 0.0;
    season_sin_coef = 0.0;
    season_log_norm = 0.0;

    season_factor_grid = rep_vector(1.0, G);
    seasonal_baseline_integral = T_end;
  }
}


model {

  real log_lik;
  matrix[D, D] S;
  log_lik = 0.0;
  S = rep_matrix(0.0, D, D);

  // Priors
  mu ~ gamma(mu_prior_shape, mu_prior_rate);
  to_vector(alpha) ~ beta(alpha_prior_a, alpha_prior_b);
  to_vector(beta) ~ gamma(beta_prior_shape, beta_prior_rate);

  if (use_seasonality == 1) {
    season_coef ~ normal(0,season_sd);
  }


  // 1. Baseline compensator -----------------------------------
  // Seasonal: mu_d * integral s(t) dt
  // Constant: mu_d * T_end

  for (d in 1:D) {
    log_lik -= mu[d] * seasonal_baseline_integral;
  }


  // 2. Excitation compensator ---------------------------------
  log_lik +=
    reduce_sum(partial_compensator, types, 1, times, marks, alpha, beta, T_endD);


  // 3. Recursive event log-intensity --------------------------
  for (i in 1:N) {

    int u;
    real lambda_at_t;
    real season_i;
    u = types[i];

    // Decay excitation accumulated from previous events -------

    if (i > 1) {

      real dt;

      dt = times[i] - times[i - 1];

      for (d in 1:D) {

        for (j in 1:D) {
          S[d, j] *=exp(-beta[d, j] * dt);
        }
      }
    }


    // Baseline multiplier -------------------------------------

    if (use_seasonality == 1) {
      season_i =
        exp(season_cos_coef * season_cos_event[i] + season_sin_coef * season_sin_event[i] - season_log_norm);
    }

    else {
      season_i = 1.0;
    }


    // Intensity for observed event type ----------------

    lambda_at_t = mu[u] * season_i;

    for (j in 1:D) {
      lambda_at_t += S[u, j];
    }

    log_lik += marks[i] * log( lambda_at_t + 1e-12);

    // Add the current event to future excitation
    for (d in 1:D) {
      S[d, u] += alpha[d, u] * beta[d, u] * marks[i];
    }
  }

  target += log_lik;
}


generated quantities {

  // Branching and decay summaries -------
  matrix[D, D] branching_ratio;
  matrix[D, D] half_life_days;
  vector[D] parent_total_offspring;
  vector[D] child_total_inflow;
  real total_baseline_compensator;

  // Branching matrix --------------------
  branching_ratio = alpha;


  // Half-life ---------------------------
  for (d in 1:D) {
    for (j in 1:D) {
      half_life_days[d, j] = log(2.0) / beta[d, j];
    }
  }

  // Total offspring generated by each parent category

  for (j in 1:D) {

    parent_total_offspring[j] = 0.0;

    for (d in 1:D) {

      parent_total_offspring[j] += alpha[d, j];
    }
  }


  // Total excitation received by each child category

  for (d in 1:D) {

    child_total_inflow[d] = 0.0;

    for (j in 1:D) {
      child_total_inflow[d] += alpha[d, j];
    }
  }


  // Baseline compensator
  total_baseline_compensator = sum(mu) * seasonal_baseline_integral;
}
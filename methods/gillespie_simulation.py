# %% markdown
# # Stochastic Simulation Algorithm - Gillespie (-Doob)
#
# This algorithm simulate a process ruled by a master equation with Monte-Carlo stochastic methods
# There are basically 4 essential steps:
#
# >* **Initialization:** Give all the initial occupation numbers for every reactant, define what are the chemical reactions and their kinetic constants
# >* **Monte-Carlo:** go to the casino...which of the defined reactions is going to win and what time you have to wait to make it winning
# >* **Update:**  modify the system updating the occupation numbers and update the time
# >* **Repeat:** from step 2 until we reach the stopping condition (typically the global simulation time, but it could also be
# a minimal amount of reactant)
#
# Then we can repeat many times the simulation to plot distributions and have an idea of time variability of the results
#
# ![Gillespie Trajectories](http://www.theosysbio.bio.ic.ac.uk/wp-content/uploads/2011/07/example3a.png)
#
# ## About the algorithm
#
#    Let $\lambda_i(s)$ be the propensity functions of each individual chemical reaction of the system. They depend on the **state** $s$ of the system, but not on **time**.
#
# Each reaction has a constant probability to take place for every time interval, therefore the probability that in a given time interval $\Delta t$, if the system is in a state $s$ the reaction $i$ **does not** take place is an exponential:
#
# ### $ p_i(s,\Delta t)=e^{-\lambda_i(s)\Delta t}$
#
#
# from that it follows that the sojourn time $\tau$ (time interval of no reactions) is given by the sum of all propensity functions (time constants of the exponential):
#
# ### $p(s,\tau)=e^{-\sum_i \lambda_i(s)\tau}$
#
# Now we just have to choose what reaction takes place.
# We know that after a time $\tau$ a reaction took place, by definition of $\tau$, so we have to draw the most likely reaction that took place, given the state of the system.
#
# Intuitively, the probability that a single reaction $i$ took place is proportional the time that the system spends in that reaction for a given time interval, so to its propensity function $\lambda_i(s)$:
#
# ### $p_i = \frac{\lambda_i(s)}{\sum_i \lambda_i(s)}$
# %% markdown
# ## Interconversion reactions
#
# We choose a simple system to see if the algorithm works properly, in this way we are able to compare it with the analytical solution of the master equation.
#
# The minimal model chosen is an interconversion reaction, where we fixed a number of molecules $N$ that can be in each of the two states A and B. A single particle can go from a state to the other with a constant time probability, therefore on average for a large number of particles we have an exponential decay, which has a constant time probability for going from A to B is given by the product of the number of particles in state A times the kinetic constant $k_a$.
#
# It can be proven that for this system (write the master Eq.) the stationary solution is the binomial:
# ### $P_{n_A}^s = \binom {n_A} {N} (\frac{k_a}{k_a + k_b})^{n_A} (\frac{k_b}{k_a + k_b})^{n_B}$
#
# which has an average:
# ### $ \langle{n_A}\rangle = N \frac{k_a}{k_a + k_b}$
#
# The typical relaxation time to the stationary state can also be computed and is:
# ### $\tau_c = \frac{1}{k_a + k_b} $
#
# ## Definition of state of the system
#
# Let us generate a 2 states system, with states A and B.
#
# In python it is represented with a class **State**, containing all the necessary information about the state of the system:
# the current number of molecules in each state (**A** and **B**), the sojourn time untill next reaction (**dt**) and the global time from the beginning **t**.
# %% codecell


# %% markdown
## State_c class
# We define a new class State_c for the rates defined with respect to concentrations, instead of
# the number of particles. In this way we set the parameters $k_1$ and $k_2$
# and the bifurcation point should not depend on $N$ anymore.
# %%

class State(namedtuple('State',["A", "B", "C", "dt", "t"])):

    def __add__(self, s2):
        """adding the states"""
        return State_c(self.A+s2.A,
                     self.B+s2.B,
                     self.C+s2.C,
                     self.dt+s2.dt,
                     self.t+s2.t)

    def pi_BA(self, k_1, k_2, N):
        """Transition rate from A to B"""
        return self.A*k_2/(N*k_1*k_2 + k_1*self.B + k_2*self.A)

    def pi_AB(self, k_1, k_2, N):
        """Transition rate from B to A"""
        return self.B*k_1/(N*k_1*k_2 + k_1*self.B + k_2*self.C)

    def pi_CB(self,k_1, k_2, N):
        """Transition rate from B to C"""
        return self.B*k_1/(N*k_1*k_2 + k_1*self.B + k_2*self.A)

    def pi_BC(self,k_1, k_2, N):
        """Transition rate from C to B"""
        return self.C*k_2/(N*k_1*k_2 + k_2*self.C + k_1*self.B)


# %% codecell
# %% markdown
# ## The definition of the chemical reactions
#
# Here, we can have 4 different reactions:
#
# * A $\Rightarrow$ B
# * B $\Rightarrow$ A
#
# The effect of each of them on the system is the unit change of the number of particles in each state (it increases in one and decreases in the other).
#
# To define a reaction we use the State class to introduce the effect that the reaction has on the system, and a function for each reaction returning the rate of the equation, given the system state
# Parameters for mono stable state are: $K_1 = 3, K_2 = 50 $ and for bistable state $K_1 = 3, K_2 = 0.5$
# %% codecell


mod_R1 = State(-1,1,0,0,0)
mod_R2 = State(1,-1,0,0,0)
mod_R3 = State(0,-1,1,0,0)
mod_R4 = State(0,1,-1,0,0)

# %% codecell
# %% markdown
# Let us check if it works as expected
#
# Let us now simulate what reaction is going to take place: if the random number is smaller than the propensity function for the first reaction normalized by $\lambda_{tot}$, the first one takes place, otherwise the second one.
#
# Let us now draw the sojourn time. As explained previously we just have to draw a random number from the exponential distribution with a constant rate $\lambda_{tot}$, depending on the actual (before the reaction step) system state.
# ## The time-step function
#
# This function is the heart of the simulation. It has 2 parts
#
# * It draws the occurring reaction and the sojourn time and it obtains its effects on the system
# * It creates the new state and return both (the old one and the updated one)
# %% codecell

def step(state):
    lambda_tot = k_R1(state)+k_R2(state)+k_R3(state)+k_R4(state)
    kin  = np.add.accumulate([ f(state) for f in kinetics ])
    lambda_tot = kin[-1]
    kin /= lambda_tot

    idx = np.searchsorted(kin, np.random.rand())
    delta_s = modific[idx]

    dt = np.random.exponential(1./lambda_tot)
    state = state._replace(dt=dt)
    delta_s = delta_s._replace(t=dt)

    new_state = state + delta_s
    return state,new_state


# %% codecell
# %% markdown
# ## Run function for the simulation
# %% codecell

def run(old_s,t_end):
    while old_s.t < t_end:
        old_s,new_s = step(old_s)
        yield old_s
        old_s = new_s



# %% markdown
## Simulation using class State_c
# Here we the rates depending on the parameters $k_i$, which do not depend on $N$
#
# %%

t_term = 1000
t_max_steps = 20000.
N_min = 5
N_max = 205
N_step = 10
k_1 = 0.1
k_2 = 1
v_2 = v_3 = 3.04  #put integer value
v_1 = v_4 = 1.
n_steps = 150
#N = 105
# %%
for (v_2) in (np.linspace(1.5,2.49,100)):
    v_3 = v_2
    for N in np.arange(N_min, N_max+1, N_step):
        steps = []
        k_R1 = lambda s: v_1*s.pi_BA(k_1, k_2, N)
        k_R2 = lambda s: v_2*s.pi_AB(k_1, k_2, N)
        k_R3 = lambda s: v_3*s.pi_CB(k_1, k_2, N)
        k_R4 = lambda s: v_4*s.pi_BC(k_1, k_2, N)
        kinetics = [k_R1, k_R2, k_R3, k_R4]
        modific = [mod_R1, mod_R2, mod_R3, mod_R4]
        for j in np.arange(n_steps):
            rng = np.random.default_rng()
            alpha = [1,1,1]
            dir_var = N*dirichlet.rvs(alpha, size=1, random_state = rng)  # initializa with Dirichlet distribution
            n_A, n_C = np.int_(np.round(dir_var[0,:2]))
            #n_A = n_C = rng.integers(0, round(N/2), endpoint=True)  #initialize with random uniform n_A = n_C
            n_B = N - n_A - n_C
            current = State(n_A, n_B, n_C, 0, 0)
            steps_temp = [i for i in run(current, t_max_steps)]
            steps.extend((A, B, C, dt, t) for A, B, C, dt, t in steps_temp if t > t_term)
        A, B, C, delta_t, time = zip(*steps)
        p_AC_t = np.zeros((N+1, N+1))
        p_AC = np.zeros((N+1, N+1))
        p_AB_t = np.zeros((N+1, N+1))
        #distributionA = Counter()
        distribution_times = Counter()
        for A, B, C, dt, t in ((A, B, C, dt, t) for A, B, C, dt, t in steps):
            #distributionA[(A, B, C)] += 1
            distribution_times[(A, B, C)]+=dt
        #for (A, B, C), val in distributionA.items():
        #    p_AC[A, C] = val
        for (A, B, C), val in distribution_times.items():
            p_AB_t[A, B] = val
            p_AC_t[A, C] = val
        p_AB_t = p_AB_t/np.sum(p_AB_t)
        p_AC_t = p_AC_t/np.sum(p_AC_t)
        p_AC = p_AC/np.sum(p_AC)
        namefile = os.path.join(path, "bistable", "{}_bistable_GIL_v2_{}".format(N, f"{v_2:.2f}")) if v_2 >= \
                   2.6 else os.path.join(path, "monostable", "{}_monostable_GIL_v2_{}".format(N, f"{v_2:.2f}"))
        #np.savetxt(namefile + "_AB_time_vel.txt", p_AB_t.transpose()) #label AC for p_AC and AB for p_AB
        np.savetxt(namefile + "_AC_time_vel_k.txt", p_AC_t.transpose())
        #np.savetxt(namefile + "_AC_vel_k.txt", p_AC.transpose())

# %%

# %% markdown
#
#   > **STEP 0.**   At time $t = 0$ assign the initial values to the variables
# $n_1$, $n_2$, ... and to the parameters $k_i$. Calculate the quantities $\lambda_i(s)$
# which in practice determine $P(\tau, i)$. One can also define the time of
# observation $t_1 < t_2 < ...$ and the stopping time $t_s$.
#
#    >**STEP 1.** Make use of a dedicated Monte Carlo technique to generate
# a random pair $(\tau, i)$ (sojourn time and reaction), which obeys to the joint probability density
# function $P(\tau, i)$.
#
#    >**STEP 2.** Make use of the values as generated above to advance the
# system in time by a quantity $\tau$, while adjusting the values of the
# population sizes $n_i$ implicated in the selected reaction $i$. After this
# operation is being taken to completion, calculate again the quantities
# $\lambda_i(s)$ for those reactions that have experienced a change in the
# chemicals amount.
#
#    >**STEP 3.** If time $t$ is less than $t_s$ or if there are no reactants left into
# the system ($\lambda_i(s) = 0$) stop the simulations. In our case there is a feedback, so $\lambda_i(s) \neq 0$ for any $t$. Otherwise, start again from
# STEP 1.
#
#

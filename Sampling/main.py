import numpy
import numpy as np
import matplotlib.pyplot as plt

defaultVar = 1.0

def get_reward(Q_STAR,a,var):
    return float(rng.normal(Q_star[a],var,1))


nb_actions = 5
actions = np.arange(nb_actions)  #5 random true values
print(actions)

rng = np.random.default_rng()
Q_star = rng.normal(0,1,nb_actions)

print(Q_star)

plt.figure(figsize=(10,5))
plt.plot(Q_star)
plt.figure(figsize=(10,5))
plt.bar(actions,Q_star)
plt.ylabel("Q")
plt.xlabel("Actions")
plt.show()

nb_samples = 10
rewards = np.zeros((nb_actions,nb_samples))

for a in actions:
    for j in range(nb_samples):
        rewards[a,j] = get_reward(Q_star,a,1.0)

Q_t = np.mean(rewards,1)

plt.figure(10,5)
plt.bar(actions,Q_t)
plt.xlabel("Actions")
plt.ylabel("$Q_t(a)$")
plt.show()

difference = numpy.zeros(nb_actions)
for i in difference:
    difference[i] = Q_star[i] - Q_t[i]

plt.figure(10,5)
plt.bar(actions,difference)
plt.xlabel("Actions Difference")
plt.ylabel("$Q_t(a)$")
plt.show()



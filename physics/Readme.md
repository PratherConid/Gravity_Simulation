__Thermometer__
* We need to find out a way to measure the temperature of substance in a microscopic setting.
* The first approach is the __ideal gas internal energy gauge:__ ```IdealGasThermometer```. We let a box of ideal gas particles come in thermal contact with the substance being measured. After thermal equilibrium is established, we measure the internal energy of the ideal gas and use the relation
  $$U = \frac{3}{2}kT$$
  to obtain the value of $kT$
* The second approach is the __kinetic gauge:__ __TODO__. Consider a hypothetical particle type $A$. The interaction between $A$ and any other particle (including particles of type $A$) is equivalent to two balls of radius $r$ colliding ($r$ is small enough), which means that in type $A$ particle's "view", all particles are rigid spheres of radius $r$. We also require that the mass $m_A$ of particle $A$ is small compared to the particles being measured. Now, let a set of particles of type $A$ come in thermal contact with the substance being measured, and denote the particle of the substance being measured as particle type $B$. Let the number of particle $A$ be large enough so that the population density of $B$ is negligible compared to the population density of $A$. In this way, type $A$ particles' velocity follow the Maxwell-Boltzmann distribution
  $$f(v) dv = \left(\frac{m_A}{2\pi kT}\right)^{3/2}4\pi v^2 \exp\left(- \frac{m_Av^2}{2kT}\right)dv$$
  Now, let type $B$ particles' velocity distribution follow the distribution
  $$g(v) dv$$
  __TODO__
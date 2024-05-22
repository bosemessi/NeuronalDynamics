# End of the chapter questions. Question 1 has been done on Mathematica; rest are just easier to type up on Latex/Markdown as they are all analytical stuff.


## Qs 2. 

$` d^2u/dt^2 = (df/du) (du/dt) `$ For this to be zero at a point where $` du/dt `$ is not, $` df/du = 0`$, which defines the rheobase voltage value.


## Qs 3.

### a. 

$` g_{eff} = \sum_k g_k `$ and $` E_{eff} = \left( \sum_k g_k E_k \right)/ \left(\sum_k g_k \right) `$

### b. 

$` m_0(u) = \frac{1}{1 + e^{-\beta (u - \theta_{act})}} = \frac{e^{\beta(u - \theta_{act})}}{1 + e^{\beta(u-\theta_{act})}} `$. In the given regime
$` u < \theta_{act} - \beta^{-1}  `$, this is equivalent to $` \beta (u - \theta_{act}) < -1 `$, so the expoential term is small and can be ignored in the denominator which means $` m_0(u) \approx e^{\beta(u - \theta_{act})} `$.

### c. 

Plugging in above gives a term proportional to $` e^{3\beta(u - \theta_{act})} `$ multiplied by a bunch of constants.

### d. 

Combine the potassium and leak currents like in part a to form an effective term. Ignoring external current ($`I = 0`$), we get

$$
    C \frac{du}{dt} = -g_{eff} (u - E_{eff}) - g_{Na} h_{rest} (u_{rest} - E_{Na}) e^{3\beta(u - \theta_{act})}
$$

Therefore, $` u_{rest} = E_{eff}`$, $` \Delta_T = (3 \beta)^{-1} `$, $` \tau = g_{eff}/C = - g_{Na} h_{rest} (u_{rest} - E_{Na}) / C \Delta_T  `$ (the last equality comes from the definition of rheobase voltage -> same methodology as Question 2). 




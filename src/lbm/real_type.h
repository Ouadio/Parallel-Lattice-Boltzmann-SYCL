#pragma once

/**
 * \typedef real_t (alias to float or double)
 */
#ifdef USE_DOUBLE
using real_t = double;
#else
using real_t = float;
#endif // USE_DOUBLE
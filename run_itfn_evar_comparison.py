from itfs import run_itfn_vs_evar_comparison

if __name__ == '__main__':
    # Full run: quick_test=False uses paper parameters (pop=180, gen=2000, runs=30)
    # which takes ~30+ hours.  We use a substantive middle-ground instead:
    # pop=80, gen=500, runs=5 — enough for convergence, ~30-45 min total.
    summary = run_itfn_vs_evar_comparison(
        markets=('nse', 'nyse'),
        quick_test=False,
        verbose=True,
    )
    print('\nDone. Review the table above to compare ITFN Model I vs EVaR Model IV.')

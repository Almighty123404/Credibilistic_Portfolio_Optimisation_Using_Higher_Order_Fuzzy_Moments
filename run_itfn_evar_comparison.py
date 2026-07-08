from itfn import run_itfn_vs_evar_comparison

if __name__ == '__main__':
    # 3-way comparison: ITFN-SV vs ITFN-EVaR vs CTFN-EVaR
    #
    # quick_test=True  : pop=30, gen=50, runs=2  (~3-5 min)
    # quick_test=False : pop=80, gen=250, runs=5 (~45-90 min, recommended for final results)
    summary = run_itfn_vs_evar_comparison(
        markets=('nse', 'nyse'),
        quick_test=False,
        verbose=True,
    )
    print('\nDone. Review the 3-way table above:')
    print('  ITFN Model I  (CITFN + Semivariance)')
    print('  ITFN Model IV (CITFN + EVaR)         <- new')
    print('  EVaR Model IV (CTFN  + EVaR)         <- baseline')

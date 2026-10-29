# Review 3 Summary (data: Aug 22 - Oct 29, 2026)

**Data:** 7,922 raw rows, 7,172 cleaned, 40 products, 7 subcategories, 69 days, 3 scrapes/day. All four festival windows covered, including Navratri + Dussehra (Oct 11-20).

## EDA: when do prices change?
Only 3.4% of observations show a price move (>0.5%).
- Festival 3.56% vs non-festival 3.22%: almost no difference overall.
- Navratri + Dussehra is the exception: 5.90%. Ganesh Chaturthi 2.61%, Janmashtami 2.21%, Onam/Raksha Bandhan 2.35% are all below the non-festival rate.
- Weekend 2.62% vs weekday 3.68%: prices change LESS on weekends.
- Midnight run 5.39%, afternoon 3.21%, morning 1.58%: changes are applied overnight.
- Most volatile subcategories: Beauty 6.2%, Fashion/Footwear 4.5%, Television 4.2%.

## ML
Regression (next_price_change_pct), full data, 6,322 rows, chronological 80/20:
- RF RMSE 3.31 / MAE 0.64 / R2 0.04; XGB RMSE 3.28 / MAE 0.69 / R2 0.06. R2 near zero: price jumps are not predictable from this feature set.
- Filtered to real changes (272 rows, up from 87): baseline RMSE 14.59; RF 12.87 (-11.8%); XGB 12.42 (-14.9%).
Classifier (will the next observation show a change?), test positives 4.8%:
- RF AUC 0.708, XGB AUC 0.690 (baseline 0.5). F1 only 0.04 / 0.12. Ranking is better than chance, but exact hits are poor.
- Top features: rolling mean/std, discount, mrp. Calendar features matter less.

## Deceptive discounts (rule-based)
- A. Displayed vs computed discount: 0 mismatches in 6,438 rows. Flipkart's displayed % equals (MRP-SP)/MRP. This check carries no signal.
- B. MRP raised while SP rose less than half as much: 2 rows, 2 products.
- C. Festival discount >=5pp above baseline without a lower SP: 25 of 3,010 festival rows (0.83%), all one product (PHILIPS air fryer HD9252).
- Festival SP vs baseline is lower in every subcategory: Beauty -3.9%, Fashion -3.2%, Home Appliance -1.25%, Laptop -1.24%.
- Conclusion: no widespread discount inflation in this 40-product sample. One product shows it.

## Limitations
40 products, 10 weeks, 3.4% change rate, one season of festivals. Results are directional.

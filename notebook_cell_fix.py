# Fixed Cell 40 - Calculate optimal trading labels using V2 optimizer
# Replace the existing cell in train_dual_transformers.ipynb with this code

print("\n3. Calculating optimal trading labels...")
print("   (Applying business rules to create training targets)\n")

optimal_decisions = []
optimal_quantities = []
optimal_prices = []

# Initialize battery at 35% SoC (room to buy)
initial_soc = 0.35
current_battery_charge = 40.0 * initial_soc

for i in tqdm(range(len(consumption_predictions))):
    # Battery state - track it ourselves instead of using old data
    battery_state = {
        "current_charge_kwh": current_battery_charge,
        "capacity_kwh": 40.0,
        "min_soc": 0.20,
        "max_soc": 1.0,  # Changed from 0.90
        "max_charge_rate_kw": 10.0,
        "max_discharge_rate_kw": 8.0,
        "efficiency": 0.95,
    }

    # Get pricing data for next 48 intervals
    price_data = (
        pricing_df["price_per_kwh"].values[i : i + 48]
        if i + 48 <= len(pricing_df)
        else pricing_df["price_per_kwh"].values[-48:]
    )

    # Calculate optimal trading decision using V2 optimizer
    labels = calculate_optimal_trading_decisions(
        predicted_consumption=consumption_predictions[i],
        actual_prices=price_data,
        battery_state=battery_state,
        household_price_kwh=0.27,
        buy_threshold_mwh=20.0,
        sell_threshold_mwh=40.0,
        min_soc_for_sell=0.25,  # New parameter
        target_soc_on_buy=0.90,  # New parameter
    )

    optimal_decisions.append(labels["optimal_decisions"])
    optimal_quantities.append(labels["optimal_quantities"])
    optimal_prices.append(price_data[0])

    # Update battery charge based on first decision in the 48-interval window
    # This simulates the battery evolving over time
    decision = labels["optimal_decisions"][0]
    quantity = labels["optimal_quantities"][0]

    if decision == 0:  # Buy
        current_battery_charge += quantity * battery_state["efficiency"]
    elif decision == 2:  # Sell
        current_battery_charge -= quantity

    # Subtract consumption (this is critical!)
    current_battery_charge -= consumption_predictions[i][0]

    # Clip to bounds
    current_soc = current_battery_charge / 40.0
    current_soc = np.clip(current_soc, 0.20, 1.0)
    current_battery_charge = 40.0 * current_soc

optimal_decisions = np.array(optimal_decisions)
optimal_quantities = np.array(optimal_quantities)
optimal_prices = np.array(optimal_prices)

print(f"\n   ✓ Calculated {len(optimal_decisions):,} optimal trading labels")
print(f"   ✓ Decision distribution:")
unique, counts = np.unique(optimal_decisions[:, 0], return_counts=True)
for decision, count in zip(unique, counts):
    decision_name = ["Buy", "Hold", "Sell"][int(decision)]
    print(f"      - {decision_name}: {count} ({count/len(optimal_decisions)*100:.1f}%)")

# Show final battery state
print(f"   ✓ Final battery SoC: {current_soc*100:.1f}%")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
customer=pd.read_csv('-commerce Sales Analytics System/olist_customers_dataset.csv',encoding='latin1')
order=pd.read_csv('-commerce Sales Analytics System/olist_orders_dataset.csv',encoding='latin1')
seller=pd.read_csv('-commerce Sales Analytics System/olist_sellers_dataset.csv',encoding='latin1')
location=pd.read_csv('-commerce Sales Analytics System/olist_geolocation_dataset.csv',encoding='latin1')
payment=pd.read_csv('-commerce Sales Analytics System/olist_order_payments_dataset.csv',encoding='latin1')
reviews=pd.read_csv('-commerce Sales Analytics System/olist_order_reviews_dataset.csv',encoding='latin1')
products=pd.read_csv('-commerce Sales Analytics System/olist_products_dataset.csv',encoding='latin1')
category_name=pd.read_csv('-commerce Sales Analytics System/product_category_name_translation.csv',encoding='latin1')
items=pd.read_csv('-commerce Sales Analytics System/olist_order_items_dataset.csv',encoding='latin1')
'''print('customer------------------------------')
print(customer.head(1))
print('order------------------------------')
print(order.head(1))
print('seller------------------------------')
print(seller.head(1))
print('location------------------------------')
print(location.head(1))
print('payment------------------------------')
print(payment.head(1))
print('reviews------------------------------')
print(reviews.head(1))
print('products------------------------------')
print(products.head(1))
print('category_name------------------------------')
print(category_name.head(1))
print('items------------------------------')
print(items.head(1))
'''

#replacing null values
order.fillna('unknown',inplace=True)
reviews.fillna('unknown',inplace=True)
products['product_id']=products['product_id'].fillna('unknown',inplace=True)
products['product_category_name']=products['product_category_name'].fillna('unknown',inplace=True)
products['product_name_lenght']=products['product_name_lenght'].interpolate(method='linear')
products['product_description_lenght']=products['product_description_lenght'].interpolate(method='linear')
products['product_photos_qty']=products['product_photos_qty'].interpolate(method='linear')
products['product_weight_g']=products['product_weight_g'].interpolate(method='linear')
products['product_length_cm']=products['product_length_cm'].interpolate(method='linear')
products['product_height_cm']=products['product_height_cm'].interpolate(method='linear')
products['product_width_cm']=products['product_width_cm'].interpolate(method='linear')

#Dataset Overview
print('----------Dataset Overview----------')
total_no_of_order=order['order_id'].nunique()
print('\nTotal number of orders:',total_no_of_order)

total_no_of_customer=customer['customer_id'].nunique()
print('\nTotal number of customer:',total_no_of_customer)

total_no_of_products=products['product_id'].nunique()
print('\nTotal number of products:',total_no_of_products)

total_no_of_sellers=seller['seller_id'].nunique()
print('\nTotal number of sellers:',total_no_of_sellers)

total_no_of_cities=location['geolocation_city'].nunique()
print('\nTotal number of cities:',total_no_of_cities)

total_no_of_states=location['geolocation_state'].nunique()
print('\nTotal number of states:',total_no_of_states)

#Revenue Analysis
print('\n\n----------Revenue Analysis----------')

total_revenue=payment['payment_value'].sum()
print('\nTotal revenue generated:',total_revenue)

revenue_mean=payment['payment_value'].mean()
print('\nAverage order value:',revenue_mean)
#convert it in datetime
order['order_purchase_timestamp']=pd.to_datetime(order['order_purchase_timestamp'])
order['year_month']=order['order_purchase_timestamp'].dt.to_period('M')
revenue_data=pd.merge(order,payment,on='order_id')
monthly_revenue=revenue_data.groupby('year_month')['payment_value'].sum()
highest_monthly_revenue=monthly_revenue.max()
highest_month=monthly_revenue.idxmax()
print('\nHighest revenue month:',highest_month,':',highest_monthly_revenue)
lowest_monthly_revenue=monthly_revenue.min()
lowest_month=monthly_revenue.idxmin()
print("\nLowest revenue month:",lowest_month,":",lowest_monthly_revenue)
growth_rate=monthly_revenue.pct_change()*100
print('\nMean growth rate:',growth_rate.mean())
#order analysis
print('\n----------Order Analysis----------')
print('\n',payment['payment_type'].value_counts())


#customer ayalysis
print('\n\n----------Customer Analysis----------')
print('\nunique customers:',order['customer_id'].nunique())
a=pd.merge(payment,order,on='order_id')
b=pd.merge(a,customer,on='customer_id')
b.sort_values(by='payment_value',ascending=False,inplace=True)

print('\nTop 5 customers by spending')
print(b[['payment_value','customer_id']].head(5))
print('Average customer spending:',payment['payment_value'].mean())

x=customer.groupby('customer_state')['customer_id'].count().sort_values(ascending=False)
print('\nTop 5 customer distributions by state:',x.head())

print('\n\n----------Product Analysis----------')

print('\nTop 5 sold category:')
print(products.groupby('product_category_name')['product_category_name'].count().sort_values(ascending=False).head())

print('\nLeast 5 sold category:')
print(products.groupby('product_category_name')['product_category_name'].count().sort_values(ascending=True).head())

print('\nTop 5 product category distribution')
print(products.groupby('product_category_name')['product_id'].count().sort_values(ascending=False).head())


print('\n\n----------Seller Analysis----------')
print('\nTotal sellers:',seller['seller_id'].nunique())

print('\nTop 5 sellers by revenue:',items.groupby('seller_id')['price'].sum().sort_values(ascending=False).head())

print('\nTop 5 sellers by order:',items.groupby('seller_id')['order_id'].count().sort_values(ascending=False).head())

print('\nTop 5 seller distribution by state:',seller.groupby('seller_state')['seller_id'].count().sort_values(ascending=False).head())

print('\nBest performing seller:',items.groupby('seller_id')['price'].sum().sort_values(ascending=False).head(1))

print('\n\n----------Payment Analysis----------')
print('\nMost used payment method:',payment.groupby('payment_type')['payment_type'].count().sort_values(ascending=False).head(1))

distribution=pd.merge(order,payment,on='order_id')
payment_distribution=pd.merge(distribution,customer,on='customer_id')
print('\nTop 5 payment method distribution by state:',payment_distribution.groupby('customer_state')['payment_type'].value_counts().sort_values(ascending=False).head())

print('\nAverage payment value:',payment['payment_value'].mean())
print('\nAverage installment analysis:',payment['payment_sequential'].mean())

print('\n\n----------Review Analysis----------')
print('\nAverage review score:',reviews['review_score'].mean())

score=pd.merge(reviews,order,on='order_id')
total_review_score=pd.merge(score,customer,on='customer_id')
print('\nTop 5 review score by state distribution:',total_review_score.groupby('customer_state')['review_score'].mean().sort_values(ascending=False).head())

rate=pd.merge(items,reviews,on='order_id')
rating=pd.merge(rate,products,on='product_id')

best_rated_category=rating.groupby('product_category_name')['review_score'].mean().sort_values(ascending=False).head(1)
print('\nBest rated category:',best_rated_category)

least_rated_category=rating.groupby('product_category_name')['review_score'].mean().sort_values(ascending=True).head(1)
print('\nLeast rated category:',least_rated_category)

print('\n\n----------Geographic Analysis----------')
q=pd.merge(order,customer,on='customer_id')
w=pd.merge(q,payment,on='order_id')
print('\nTop state by revenue:',w.groupby('customer_state')['payment_value'].sum().sort_values(ascending=False).head(1))
print('\nLeast state by revenue:',w.groupby('customer_state')['payment_value'].sum().sort_values(ascending=True).head(1))

print('\nTop state by orders:',w.groupby('customer_state')['order_id'].count().sort_values(ascending=False).head(1))
print('\nLeast state by orders:',w.groupby('customer_state')['order_id'].count().sort_values(ascending=True).head(1))


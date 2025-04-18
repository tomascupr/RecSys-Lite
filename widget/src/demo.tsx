import React from 'react';
import ReactDOM from 'react-dom/client';
import { RecommendationCarousel } from './RecommendationCarousel';

// Mock data for demonstration
const mockRecommendations = [
  {
    item_id: 'item1',
    score: 0.95,
    title: 'Wireless Headphones',
    category: 'Electronics',
    brand: 'SoundMax',
    price: 89.99,
    image_url: 'https://images.unsplash.com/photo-1606813907291-d86efa9b94db?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=300&h=300'
  },
  {
    item_id: 'item2',
    score: 0.89,
    title: 'Smart Watch',
    category: 'Electronics',
    brand: 'TechWear',
    price: 199.99,
    image_url: 'https://images.unsplash.com/photo-1523275335684-37898b6baf30?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=300&h=300'
  },
  {
    item_id: 'item3',
    score: 0.85,
    title: 'Organic Cotton T-shirt',
    category: 'Clothing',
    brand: 'EcoWear',
    price: 29.99,
    image_url: 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=300&h=300'
  },
  {
    item_id: 'item4',
    score: 0.82,
    title: 'Ceramic Coffee Mug',
    category: 'Kitchen',
    brand: 'HomeGoods',
    price: 14.99,
    image_url: 'https://images.unsplash.com/photo-1514228742587-6b1558fcca3d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=300&h=300'
  },
  {
    item_id: 'item5',
    score: 0.78,
    title: 'Leather Wallet',
    category: 'Accessories',
    brand: 'LeatherCraft',
    price: 49.99,
    image_url: 'https://images.unsplash.com/photo-1605348532760-6753d2c43329?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=300&h=300'
  },
];

function Demo() {
  const [userId, setUserId] = React.useState('user123');

  return (
    <div className="container mx-auto p-8 max-w-5xl">
      <div className="flex flex-col items-center mb-8">
        <h1 className="text-3xl font-bold mb-4">RecSys-Lite Widget Demo</h1>
        <p className="text-gray-600 text-center max-w-2xl mb-6">
          This is a demonstration of the RecSys-Lite recommendation widget. It displays product recommendations in a responsive carousel.
        </p>
        
        <div className="mb-4 flex items-center">
          <label htmlFor="userId" className="mr-2 font-medium">User ID:</label>
          <input 
            type="text" 
            id="userId" 
            value={userId} 
            onChange={(e) => setUserId(e.target.value)}
            className="px-3 py-2 border rounded shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>
      
      <div className="border rounded-lg shadow-md p-6 bg-white">
        <RecommendationCarousel
          apiUrl="https://api.example.com"
          userId={userId}
          count={5}
          title="Recommended Products"
          testRecommendations={mockRecommendations}
          onItemClick={(item) => {
            alert(`Clicked on: ${item.title} (${item.item_id})`);
          }}
        />
      </div>
      
      <div className="mt-12 border-t pt-6">
        <h2 className="text-xl font-semibold mb-4">Custom Styling Example</h2>
        <div className="bg-gray-50 p-6 rounded-lg">
          <RecommendationCarousel
            apiUrl="https://api.example.com"
            userId={userId}
            count={5}
            title="Products Just For You"
            className="bg-blue-50 p-4 rounded-lg shadow"
            containerClassName="gap-4"
            cardClassName="bg-white border-blue-500 shadow-lg"
            testRecommendations={mockRecommendations}
          />
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Demo />
  </React.StrictMode>,
);
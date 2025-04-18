import React from 'react';
import type { Meta, StoryObj } from '@storybook/react';
import { RecommendationCarousel } from './RecommendationCarousel';
import './lib/globals.css';

// Mock data for stories
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
  {
    item_id: 'item6',
    score: 0.75,
    title: 'Stainless Steel Water Bottle',
    category: 'Kitchen',
    brand: 'HydroMax',
    price: 24.99,
    image_url: 'https://images.unsplash.com/photo-1602143407151-7111542de6e8?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=300&h=300'
  },
];

// Define meta for the component
const meta = {
  title: 'RecSys-Lite/RecommendationCarousel',
  component: RecommendationCarousel,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    apiUrl: { control: 'text' },
    userId: { control: 'text' },
    count: { control: { type: 'range', min: 1, max: 20 } },
    title: { control: 'text' },
    className: { control: 'text' },
    containerClassName: { control: 'text' },
    cardClassName: { control: 'text' },
    onItemClick: { action: 'clicked' },
    testRecommendations: { 
      control: 'object',
      description: 'Test recommendations for Storybook (bypasses API call)'
    },
  },
} satisfies Meta<typeof RecommendationCarousel>;

export default meta;
type Story = StoryObj<typeof meta>;

// Default story with test recommendations
export const Default: Story = {
  args: {
    apiUrl: 'https://api.example.com',
    userId: 'user123',
    count: 5,
    title: 'Recommended For You',
    testRecommendations: mockRecommendations.slice(0, 5),
  },
};

// Loading state story
export const Loading: Story = {
  args: {
    apiUrl: 'https://api.example.com',
    userId: 'user123',
    count: 5,
    title: 'Recommended For You',
    // No testRecommendations, so it will show loading state forever in Storybook
  },
};

// Error state story with custom decorator
const ErrorDecorator = (Story: React.ComponentType) => {
  React.useEffect(() => {
    // Find error state element and simulate error after a short delay
    setTimeout(() => {
      const component = document.querySelector('[data-testid="recommendation-carousel"]');
      if (component) {
        const loading = component.querySelector('.flex.justify-center.items-center');
        if (loading) {
          loading.remove();
        }
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'bg-destructive/10 border border-destructive text-destructive px-4 py-3 rounded';
        errorDiv.setAttribute('role', 'alert');
        errorDiv.setAttribute('aria-live', 'assertive');
        errorDiv.textContent = 'Error fetching recommendations: User not found';
        
        // Insert after the title
        const title = component.querySelector('h2');
        if (title && title.nextSibling) {
          component.insertBefore(errorDiv, title.nextSibling);
        } else {
          component.appendChild(errorDiv);
        }
      }
    }, 500);
  }, []);
  
  return <Story />;
};

export const Error: Story = {
  args: {
    apiUrl: 'https://api.example.com',
    userId: 'invalid-user',
    count: 5,
    title: 'Recommended For You',
  },
  decorators: [ErrorDecorator],
};

// Empty recommendations story
export const Empty: Story = {
  args: {
    apiUrl: 'https://api.example.com',
    userId: 'new-user',
    count: 5,
    title: 'Recommended For You',
    testRecommendations: [],
  },
};

// Custom styling story
export const CustomStyling: Story = {
  args: {
    apiUrl: 'https://api.example.com',
    userId: 'user123',
    count: 5,
    title: 'Products Just For You',
    className: 'bg-gray-100 p-6 rounded-lg shadow-md',
    containerClassName: 'gap-4',
    cardClassName: 'bg-white border-blue-500 shadow-lg',
    testRecommendations: mockRecommendations.slice(0, 5),
  },
};
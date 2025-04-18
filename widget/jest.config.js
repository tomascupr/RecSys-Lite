module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  moduleNameMapper: {
    // Mocks for assets and style files
    '\\.css$': '<rootDir>/src/__mocks__/styleMock.js',
    '\\.svg$': '<rootDir>/src/__mocks__/fileMock.js',
    // Mock for dependencies
    '^lucide-react$': '<rootDir>/src/__mocks__/lucide-react.js',
    '^embla-carousel-react$': '<rootDir>/src/__mocks__/embla-carousel-react.js',
    '^class-variance-authority$': '<rootDir>/src/__mocks__/class-variance-authority.js',
  },
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.stories.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts',
    '!src/__mocks__/**',
    '!src/lib/components/ui/**',
    '!src/demo.tsx',
    '!src/index.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 70,
      lines: 70,
      statements: 70,
    },
  },
};
// Mock for the class-variance-authority package
const cva = (base, config) => {
  return (props) => {
    if (!props) return base;
    
    let className = base;
    
    if (config && config.variants) {
      Object.keys(config.variants).forEach(variant => {
        if (props[variant] && config.variants[variant][props[variant]]) {
          className += ` ${config.variants[variant][props[variant]]}`;
        }
      });
    }
    
    if (props.className) {
      className += ` ${props.className}`;
    }
    
    return className;
  };
};

// Stub for TypeScript types
const VariantProps = () => ({});

module.exports = {
  cva,
  VariantProps,
};
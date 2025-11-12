const bcrypt = require('bcryptjs');

const password = 'YourSecurePassword123!'; // Change this!
const hash = bcrypt.hashSync(password, 10);

console.log('Password Hash:');
console.log(hash);
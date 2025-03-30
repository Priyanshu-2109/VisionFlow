import expresss from 'express';
import cors from 'cors';

//app config
const app = expresss();
const port = 5000;

//middleware
app.use(cors());
app.use(expresss.json());

app.get('/', (req, res) => {
  res.send('API Working...');
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});